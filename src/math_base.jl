
abstract type AbstractLocalTuple end

struct LocalTuple{T<:Tuple} <: AbstractLocalTuple
    tup::T
end

function Base.getindex(locals::LocalTuple, I::AbstractArray{<:Integer}) 
    arrtype = typeof(I)
    @show arrtype
    outarr = [Base.getindex(locals, i) for i in I]
    reshape(outarr, size(I))
end
function Base.getindex(locals::LocalTuple, i::Integer)  
    if i > 2 
        D = Differential(locals.tup[1])
        Num.(ModelingToolkit.diff2term.(ModelingToolkit.value.((D^(i - 2)).(locals.tup[2]))))
    else
        locals.tup[i]
    end
end

time(locals::LocalTuple) = locals[1]
position(locals::LocalTuple) = locals[2]
velocity(locals::LocalTuple) = locals[3]
acceleration(locals::LocalTuple) = locals[4]

function generate_generic_localtuple(qsize)
    @parameters t
    qrange = Tuple(1:qsize[i] for i in 1:length(qsize))
    @variables q[qrange...](t)
    LocalTuple((t, q))
end


function find_multiplies_to_derivative(expression)
	#@show expression
	#@show typeof(expression.val)
	#@show ModelingToolkit.isdifferential(expression.val)
	diffs = []
	premults = []
	if typeof(expression) <: SymbolicUtils.Mul
		#@show "mul!!"
		coeff = expression.coeff
		push!(premults, coeff)
		for key in keys(expression.dict)
			if ModelingToolkit.isdifferential(key)
				#@show "isa diff"
				push!(diffs, key)
			else
				#@show "isnota diff"
				push!(premults, key)
			end
			#@show key
		end
		#recursed_mul_premult, recursed_mul_deriv = recursively_find_multiplies_to_derivative(
	elseif ModelingToolkit.isdifferential(expression)
		push!(diffs, expression)
	end
	diffs, premults
end


function Lagrange_equations(Lagrangian, locals::AbstractLocalTuple)
    symbolic_L = Lagrangian(locals)
    t = time(locals)
    D = Differential(t)
    qvars = position(locals)
    qdotvars = velocity(locals)
    velocity_equations = Equation[]
    position_equations = Equation[]
	for (qvar, qdotvar) in zip(qvars, qdotvars)
		lhs_premult = MTderivative(MTderivative(symbolic_L, qdotvar), t)
		lhs_diffs, lhs_mults = find_multiplies_to_derivative(lhs_premult.val)
		rhs = MTderivative(symbolic_L, qvar) 
		lhs = lhs_premult
		if length(lhs_mults) > 0
			lhs_multiplied = prod(Num.(lhs_mults))
			rhs = rhs / lhs_multiplied
			lhs = Num(lhs_diffs[1]) 
		end
		push!(velocity_equations, lhs ~ rhs)
		push!(position_equations, D(qvar) ~ qdotvar)
	end
	ArrayPartition(velocity_equations, position_equations)
end

rdot(v1, v2) = sum(v1 .* v2)

MTderivative(O, v) = ModelingToolkit.derivative(O, v; simplify=true)

rhs(eqs) = map(eq->eq.rhs, eqs)


function ode_problem_generator(Lagrangian_pre_param, locals::AbstractLocalTuple, params)
    Lagrangian = Lagrangian_pre_param(params...)
    qdim = length(SICMModelingToolkit.position(locals))
    current_Lagrange_equations = Lagrange_equations(Lagrangian, locals)

    f_1_rhs = Num.(rhs(current_Lagrange_equations.x[1]))
    f_2_rhs = Num.(rhs(current_Lagrange_equations.x[2]))

    qs = SICMModelingToolkit.position(locals)
    vs = velocity(locals)
    #dvs = [vs; qs]
    ps = params
    iv = SICMModelingToolkit.time(locals)

    f_1_built = build_function(f_1_rhs, vs, qs, ps, iv; expression=Val{false})[2]
    f_2_built = build_function(f_2_rhs, vs, qs, ps, iv; expression=Val{false})[2]

    function generator(qv0, q0, tspan, p0; saveat=nothing)
        if isnothing(saveat)
            prob = DynamicalODEProblem{true}(f_1_built, f_2_built, qv0, q0, tspan, p0)
        else
            prob = DynamicalODEProblem{true}(f_1_built, f_2_built, qv0, q0, tspan, p0; saveat=saveat)
        end
    end
    generator
end






