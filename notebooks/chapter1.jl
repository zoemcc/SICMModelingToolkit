### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ a9afdc92-61b9-11eb-1ac4-d1b9cba169ff
using DrWatson

# ╔═╡ 98cb1be8-61bc-11eb-3652-35d883d64e97
begin
	using Pkg
end

# ╔═╡ 0d5893bc-61ba-11eb-21d6-2167c33bdeec
begin
	using ModelingToolkit, Unitful, DifferentialEquations, Optim
	using Latexify
	using BenchmarkTools, LinearAlgebra, ForwardDiff, Zygote
	using Quadrature, StaticArrays
	using Test
	using SymbolicUtils
	using SymbolicUtils: to_symbolic, substitute
	using Plots
end

# ╔═╡ f65c0428-61b9-11eb-0262-6dee6a26c858
DrWatson.@quickactivate "SICMModelingToolkit"

# ╔═╡ f00943f2-61db-11eb-025f-27e4a44b10f8
MTderivative(O, v) = ModelingToolkit.derivative(O, v; simplify=true)

# ╔═╡ 50a2ab6a-61bc-11eb-3933-890b4a8dfb64
rdot(v1, v2) = sum(v1 .* v2)

# ╔═╡ 79a3f678-61cf-11eb-1d7a-31224592dc85
###### 1.4

# ╔═╡ 1a88e708-61ba-11eb-394a-fff715bf6808
@parameters t m k

# ╔═╡ 0ffa5e60-61bb-11eb-05e1-0d2a6f66ed62
@variables x(t) y(t) z(t)

# ╔═╡ 78d740de-61dd-11eb-1863-cb24f75a575a
@variables x_t(t) y_t(t) z_t(t)

# ╔═╡ 145d5854-61bb-11eb-18db-f74e83fdde06
D = Differential(t)

# ╔═╡ 1eb34a04-61bb-11eb-0c4b-8d3e4a85d3e9
q = @SVector [x, y, z]

# ╔═╡ 96d39b7a-61bb-11eb-2170-1178df4203f0
qv = @SVector [x_t, y_t, z_t]

# ╔═╡ 4936584c-61ba-11eb-1d82-11d5fd37d2e1
L_free_particle(t,x,v) = 1//2 .* m .* rdot(v,v)

# ╔═╡ be3db274-61bb-11eb-3dc1-3bb68cc3caf8
latexify(L_free_particle(t,q,qv))

# ╔═╡ c2f96f38-61bb-11eb-1d0b-bd366c75515e
q .+ q

# ╔═╡ 69333044-61bc-11eb-0784-7d2954fff119
qeqs = q .~ @SVector [4t + 7, 3t + 5, 2t + 1] 

# ╔═╡ 37011c9e-61be-11eb-0562-3b7a61c8cd5b
qns = NonlinearSystem(qeqs, q, @SVector [t])

# ╔═╡ bc4ae90c-61be-11eb-1634-45317900a809
qfunc = eval(generate_function(qns)[2])

# ╔═╡ 48367b52-61bf-11eb-3cef-e3067a350e8b
begin
	du = zeros(3)
	qfunc(du, q, [-1.0])
end

# ╔═╡ c199edee-61bf-11eb-3e76-d7c1c181f14b
du

# ╔═╡ 2c2bb89c-61c0-11eb-1526-9fdc91a344d9
ModelingToolkit.substitute(qeqs, Dict(t=>1, x=>5t + 7))

# ╔═╡ d12126f4-61c1-11eb-3a8b-215ff2d0aae4
map(qeq->substitute(qeq, Dict(x=>5t+7, y=>3t+5, z=>2t+1)), q)

# ╔═╡ a8b2d568-61c2-11eb-300f-757381afee6a
test_path = @SVector [4t + 7, 3t + 5, 2t + 1] 

# ╔═╡ 61393e6e-61dd-11eb-1d03-950972079936
@variables x_t(t)

# ╔═╡ 31f036b6-61c2-11eb-3afa-35bdfaffbad4
function Γ(qvars, coordfuncs; numD=2)
	subdict = Dict(qvars .=> coordfuncs)
	mapsubs(vals) = map(qvar->expand_derivatives(substitute(qvar, subdict)), vals)
	subq = mapsubs(qvars)
	if numD > 0
		derivativeqs = [mapsubs((D^i).(qvars)) for i in 1:numD]
		[[t]; [subq]; derivativeqs]
	else 
		[[t]; [subq]]
	end
end

# ╔═╡ 0467956e-61dd-11eb-2604-81571591b6c3
function Γ_vars(qvars; numD=2)
	subdict = Dict(qvars .=> coordfuncs)
	mapsubs(vals) = map(qvar->expand_derivatives(substitute(qvar, subdict)), vals)
	subq = mapsubs(qvars)
	if numD > 0
		derivativeqs = [mapsubs((D^i).(qvars)) for i in 1:numD]
		[[t]; [subq]; derivativeqs]
	else 
		[[t]; [subq]]
	end
end

# ╔═╡ 7c4f9a38-61c2-11eb-0d94-bf3bf88b0875
Γ(q, test_path; numD=2)[4]

# ╔═╡ e2108a08-61c2-11eb-0247-d52e97b2ec87
function compose_L_Γ(L_i)
	function composed(qvars, coordfuncs; numD=2)
		Γ_val = Γ(qvars, coordfuncs; numD=numD)
		L_i(Γ_val...)
	end
end

# ╔═╡ eaf1913c-61c5-11eb-04ee-29014e6d0f8b
L_free_composed_test_path = compose_L_Γ(L_free_particle)(q, test_path; numD=1)

# ╔═╡ 3c424016-61c8-11eb-1c0a-d3112cf4ff55
L_free_eval(1.0, 3.0)

# ╔═╡ 947249ce-61c9-11eb-23f3-3d246bd8dc0b
function Lagrangian_action(L_i, path_func, t_1, t_2, qvars, pvars, pvals; numD=2)
	L_composed = compose_L_Γ(L_i)(qvars, path_func; numD=numD)
	L_func = build_function(L_composed, [t], pvars; expression=Val{false})
	quadprob = QuadratureProblem(L_func, t_1, t_2, pvals)
	solve(quadprob, QuadGKJL(); abstol=1e-7).u
end

# ╔═╡ 0c470b92-61ca-11eb-3cde-d5112faac779
func = Lagrangian_action(L_free_particle, test_path, 0.0, 10.0, q, [m], 3.0; numD=1)

# ╔═╡ 6737dc14-61cc-11eb-3ee1-85ccdd31619c
function make_η(ν, t_1, t_2)
	(t .- t_1) .* (t .- t_2) .* ν
end

# ╔═╡ 8b97dbfe-61cc-11eb-1545-9bc13cdd6614
example_ν = @SVector [sin(t), cos(t), t^2]

# ╔═╡ a4174b44-61cc-11eb-06cf-873f32b00f21
make_η(example_ν, 0.0, 10.0)

# ╔═╡ 6a5de2d6-61cb-11eb-0d7b-538845bef060
function varied_free_particle_action(mass, path, ν, t_1, t_2)
	η = make_η(ν, t_1, t_2)
	function per_eps(ϵ)
		perturbed_path = path .+ ϵ .* η
		Lagrangian_action(L_free_particle, perturbed_path, t_1, t_2, q, [m], mass; numD=1)
	end
end

# ╔═╡ 03cefa1a-61cd-11eb-2cba-2f7b8a9f4fc0
varied_free_action_per_ϵ = varied_free_particle_action(3.0, test_path, example_ν, 0.0, 10.0)

# ╔═╡ 37ec4df4-61cd-11eb-295f-e904a0abc895
varied_free_action_per_ϵ(0.001)

# ╔═╡ 47cbd212-61cd-11eb-1899-274b2e205ead
Optim.optimize(varied_free_action_per_ϵ, -2.0, 1.0)

# ╔═╡ ef791fe8-61ce-11eb-156e-b5b67a0e7887
#### 1.5 

# ╔═╡ 39ae9600-61d0-11eb-0aed-63749ad212d0
Differential(t)(x)

# ╔═╡ 76e0551c-61cf-11eb-26a1-7522376db177
expand_derivatives(Differential(D(x))(L_free_particle(Γ(q, q; numD=1)...)))

# ╔═╡ 079bbdd8-61d2-11eb-0092-fb7654b3b0ae
Differential(x)(L_free_particle(Γ(q, q; numD=1)...))

# ╔═╡ e66f3d66-61d0-11eb-3dbb-03372172ba86
L_free_particle(Γ(q, q; numD=1)...)

# ╔═╡ eb0fb58a-61df-11eb-1f5b-c731c5ad7ece
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

# ╔═╡ 13a053bc-61e0-11eb-0802-9d678d04e727
diffs, premults = find_multiplies_to_derivative((m * D(x)).val)

# ╔═╡ b3be1294-61e3-11eb-298a-5f6e59e7b0df
length(premults)

# ╔═╡ 172bc650-61e4-11eb-3514-156630da8818
dump(prod(premults))

# ╔═╡ 1232c1e8-61e5-11eb-1a93-e74e64faab02
typeof(prod(premults))

# ╔═╡ b0066b96-61e4-11eb-1791-0d5c9a8032e0
typeof(Num(diffs[1]))

# ╔═╡ 2bc7d170-61e5-11eb-22e7-198087098446
typeof(D(x))

# ╔═╡ c54e959a-61e0-11eb-12ac-e35a69d26a18
keys((m * x).val.dict)

# ╔═╡ 15ca6e50-61d1-11eb-1f59-1b814635c335
function Lagrange_equations(L_i, qvars, qdotvars)
	symbolic_L = L_i(t, qvars, qdotvars)
	equations = []
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
		push!(equations, lhs ~ rhs)
		push!(equations, D(qvar) ~ qdotvar)
	end
	equations
end

# ╔═╡ 61d6f634-61da-11eb-3621-4bae15be5cfe
length(x)

# ╔═╡ 45a8e8d6-61d1-11eb-2e54-97bd0832cf66
lagfree = Lagrange_equations(L_free_particle, q, qv)

# ╔═╡ c6c42192-61d1-11eb-0df7-a1300ccb1fa0
lagfree[3].lhs

# ╔═╡ 2a52fcf6-61d2-11eb-3632-217ef932b5c1
lagrangian_eqs_sys = ODESystem(lagfree, t, q, qv, [m])

# ╔═╡ 63350bcc-61d2-11eb-27c1-5b88ddbb29de
Lagrangian_eqs_sys.eq

# ╔═╡ b61ca1ce-61d2-11eb-2bb9-8be89a023b1f
lagrangian_eqs_sys.eqs

# ╔═╡ 1d6d2c86-61d3-11eb-1752-4df7966d2898
@parameters k

# ╔═╡ 10953204-61d3-11eb-23ee-bd299a2e470a
L_harmonic_spring(tvar, qvar, qdotvar) = 1//2 * m * qdotvar^2 - 1//2 * k * qvar^2

# ╔═╡ 3e7fe27c-61d3-11eb-160e-39eeba93ac65
L_harmonic_spring(t, x, D(x))

# ╔═╡ 551f58ac-61d3-11eb-0fdf-79af470d57bd
lagharm = Lagrange_equations(L_harmonic_spring, x, x_t)

# ╔═╡ 8496a978-61d3-11eb-3ef9-35e8f3f18f6a
lagharm

# ╔═╡ 9d53a218-61d3-11eb-3395-bf43539f832f
harmonic_sys = ODESystem(lagharm, t)

# ╔═╡ e56035c2-61dc-11eb-20f0-b1e5e6664dd4
harmonic_sys.eqs

# ╔═╡ 90da45a6-61e1-11eb-2f00-11bba22507d7
harmonic_sys.ps[1].name

# ╔═╡ 26dd26f4-61e2-11eb-3f53-15c38b38e1c8
harmonic_sys.states[2].f.name

# ╔═╡ 4d933d2c-61e2-11eb-0433-5bcd8219b181


# ╔═╡ d7a1a312-61d4-11eb-332a-7124267dd164
begin
	tspan = (0.0, π)
	u0 = [1.0, 0.0]
	p0 = [1.0, 3.0]
end

# ╔═╡ be6c49f4-61d4-11eb-018d-079c83d1ad64
harmonic_prob = ODEProblem(harmonic_sys, [x=>u0[1], x_t=>u0[2]], tspan, [k=>p0[1], m=>p0[2]])

# ╔═╡ d0d1dab4-61d9-11eb-3a8e-7b5d67dfaf22
harmonic_sol = solve(harmonic_prob, Tsit5())

# ╔═╡ e539f43c-61d9-11eb-0453-9d96c45fcc95
plot(harmonic_sol; dpi=350)

# ╔═╡ ac536e02-61d7-11eb-1521-193caca80dbe
testeqs = [D(x) ~ 1]

# ╔═╡ bb348136-61d7-11eb-36d8-d7b6a9f28c26
testsys = ODESystem(testeqs, t, [x], [])

# ╔═╡ cabc1f6a-61d7-11eb-3bc1-a3d75e1f8058
testfunc = ODEFunction(testsys, [x], [])

# ╔═╡ f4b54fe4-61d7-11eb-19f1-3d9a86d5e46d
testprob = ODEProblem(testsys, [0.0], (0.0, 2.0))

# ╔═╡ 43cbdb98-61d8-11eb-2c8b-fded5290943d
testsol = solve(testprob, Tsit5())

# ╔═╡ 4a27a706-61d8-11eb-3d04-c5d7323a7249
plot(testsol; dpi=350)

# ╔═╡ 82cf65a0-61d8-11eb-24b3-1b1beb77e372
testsecondordereqs = [D(D(x)) ~ 1]

# ╔═╡ 9d33eee6-61d8-11eb-2a45-41a7849fc0ca
testsecondordersys = ODESystem(testsecondordereqs)

# ╔═╡ b49bc720-61d8-11eb-2ff5-099ee4ac56c0
testfosys = ode_order_lowering(testsecondordersys)

# ╔═╡ 75b20a8e-61d9-11eb-1bbf-6331e9accaf1
testfosys.eqs

# ╔═╡ cb50bb58-61d8-11eb-210d-0bbadabc5508
testfosys.states

# ╔═╡ dcac2160-61d8-11eb-1d82-b934f5d2eae7
testfoprob = ODEProblem(testfosys, [0.0, 0.0], (0.0, 4.0))

# ╔═╡ ef86b534-61d8-11eb-1c68-630277726bf4
testfosol = solve(testfoprob, Tsit5())

# ╔═╡ f9cadcc6-61d8-11eb-3fd1-374577903088
plot(testfosol; dpi=350)

# ╔═╡ 41f7fcac-61dd-11eb-37b5-dd1b334525e5
### kepler problem
### k = μ, m = m, x = η, y = ζ

# ╔═╡ 9f9ae924-61e6-11eb-2150-2ffd405f3a1e
L_kepler(t, q, qv) = 1//2 * m * rdot(qv, qv) + k / sqrt(rdot(q, q))

# ╔═╡ f7a3b9c2-61e6-11eb-2bba-f7d1d6fe1658
L_kepler(t, [x, y], [x_t, y_t])

# ╔═╡ 045eda50-61e7-11eb-136b-47bb20bf6c6e
lagkepler = Lagrange_equations(L_kepler, [x, y], [x_t, y_t])

# ╔═╡ 2b3bcca8-61e9-11eb-204f-4d0e844680cd


# ╔═╡ 22c98e36-61e7-11eb-0389-2fcfe5f3bd11
kepler_prob = ODEProblem(ODESystem(lagkepler, t), [x=>1.0, x_t=>-1.0, y=>-1.0, y_t=>1.0], (0.0, 0.2), [k=>p0[1], m=>p0[2]])

# ╔═╡ 3c58cd92-61e9-11eb-34ec-c1b507fdffc2
kepler_sol = solve(kepler_prob, Tsit5())

# ╔═╡ 434fd42e-61e9-11eb-3178-e76dc841018f
plot(kepler_sol; dpi=350)

# ╔═╡ Cell order:
# ╠═a9afdc92-61b9-11eb-1ac4-d1b9cba169ff
# ╠═f65c0428-61b9-11eb-0262-6dee6a26c858
# ╠═98cb1be8-61bc-11eb-3652-35d883d64e97
# ╠═0d5893bc-61ba-11eb-21d6-2167c33bdeec
# ╠═f00943f2-61db-11eb-025f-27e4a44b10f8
# ╠═50a2ab6a-61bc-11eb-3933-890b4a8dfb64
# ╠═79a3f678-61cf-11eb-1d7a-31224592dc85
# ╠═1a88e708-61ba-11eb-394a-fff715bf6808
# ╠═0ffa5e60-61bb-11eb-05e1-0d2a6f66ed62
# ╠═78d740de-61dd-11eb-1863-cb24f75a575a
# ╠═145d5854-61bb-11eb-18db-f74e83fdde06
# ╠═1eb34a04-61bb-11eb-0c4b-8d3e4a85d3e9
# ╠═96d39b7a-61bb-11eb-2170-1178df4203f0
# ╠═4936584c-61ba-11eb-1d82-11d5fd37d2e1
# ╠═be3db274-61bb-11eb-3dc1-3bb68cc3caf8
# ╠═c2f96f38-61bb-11eb-1d0b-bd366c75515e
# ╠═69333044-61bc-11eb-0784-7d2954fff119
# ╠═37011c9e-61be-11eb-0562-3b7a61c8cd5b
# ╠═bc4ae90c-61be-11eb-1634-45317900a809
# ╠═48367b52-61bf-11eb-3cef-e3067a350e8b
# ╠═c199edee-61bf-11eb-3e76-d7c1c181f14b
# ╠═2c2bb89c-61c0-11eb-1526-9fdc91a344d9
# ╠═d12126f4-61c1-11eb-3a8b-215ff2d0aae4
# ╠═a8b2d568-61c2-11eb-300f-757381afee6a
# ╠═61393e6e-61dd-11eb-1d03-950972079936
# ╠═31f036b6-61c2-11eb-3afa-35bdfaffbad4
# ╠═0467956e-61dd-11eb-2604-81571591b6c3
# ╠═7c4f9a38-61c2-11eb-0d94-bf3bf88b0875
# ╠═e2108a08-61c2-11eb-0247-d52e97b2ec87
# ╠═eaf1913c-61c5-11eb-04ee-29014e6d0f8b
# ╠═3c424016-61c8-11eb-1c0a-d3112cf4ff55
# ╠═947249ce-61c9-11eb-23f3-3d246bd8dc0b
# ╠═0c470b92-61ca-11eb-3cde-d5112faac779
# ╠═6737dc14-61cc-11eb-3ee1-85ccdd31619c
# ╠═8b97dbfe-61cc-11eb-1545-9bc13cdd6614
# ╠═a4174b44-61cc-11eb-06cf-873f32b00f21
# ╠═6a5de2d6-61cb-11eb-0d7b-538845bef060
# ╠═03cefa1a-61cd-11eb-2cba-2f7b8a9f4fc0
# ╠═37ec4df4-61cd-11eb-295f-e904a0abc895
# ╠═47cbd212-61cd-11eb-1899-274b2e205ead
# ╠═ef791fe8-61ce-11eb-156e-b5b67a0e7887
# ╠═39ae9600-61d0-11eb-0aed-63749ad212d0
# ╠═76e0551c-61cf-11eb-26a1-7522376db177
# ╠═079bbdd8-61d2-11eb-0092-fb7654b3b0ae
# ╠═e66f3d66-61d0-11eb-3dbb-03372172ba86
# ╠═eb0fb58a-61df-11eb-1f5b-c731c5ad7ece
# ╠═13a053bc-61e0-11eb-0802-9d678d04e727
# ╠═b3be1294-61e3-11eb-298a-5f6e59e7b0df
# ╠═172bc650-61e4-11eb-3514-156630da8818
# ╠═1232c1e8-61e5-11eb-1a93-e74e64faab02
# ╠═b0066b96-61e4-11eb-1791-0d5c9a8032e0
# ╠═2bc7d170-61e5-11eb-22e7-198087098446
# ╠═c54e959a-61e0-11eb-12ac-e35a69d26a18
# ╠═15ca6e50-61d1-11eb-1f59-1b814635c335
# ╠═61d6f634-61da-11eb-3621-4bae15be5cfe
# ╠═45a8e8d6-61d1-11eb-2e54-97bd0832cf66
# ╠═c6c42192-61d1-11eb-0df7-a1300ccb1fa0
# ╠═2a52fcf6-61d2-11eb-3632-217ef932b5c1
# ╠═63350bcc-61d2-11eb-27c1-5b88ddbb29de
# ╠═b61ca1ce-61d2-11eb-2bb9-8be89a023b1f
# ╠═1d6d2c86-61d3-11eb-1752-4df7966d2898
# ╠═10953204-61d3-11eb-23ee-bd299a2e470a
# ╠═3e7fe27c-61d3-11eb-160e-39eeba93ac65
# ╠═551f58ac-61d3-11eb-0fdf-79af470d57bd
# ╠═8496a978-61d3-11eb-3ef9-35e8f3f18f6a
# ╠═9d53a218-61d3-11eb-3395-bf43539f832f
# ╠═e56035c2-61dc-11eb-20f0-b1e5e6664dd4
# ╠═90da45a6-61e1-11eb-2f00-11bba22507d7
# ╠═26dd26f4-61e2-11eb-3f53-15c38b38e1c8
# ╠═4d933d2c-61e2-11eb-0433-5bcd8219b181
# ╠═d7a1a312-61d4-11eb-332a-7124267dd164
# ╠═be6c49f4-61d4-11eb-018d-079c83d1ad64
# ╠═d0d1dab4-61d9-11eb-3a8e-7b5d67dfaf22
# ╠═e539f43c-61d9-11eb-0453-9d96c45fcc95
# ╠═ac536e02-61d7-11eb-1521-193caca80dbe
# ╠═bb348136-61d7-11eb-36d8-d7b6a9f28c26
# ╠═cabc1f6a-61d7-11eb-3bc1-a3d75e1f8058
# ╠═f4b54fe4-61d7-11eb-19f1-3d9a86d5e46d
# ╠═43cbdb98-61d8-11eb-2c8b-fded5290943d
# ╠═4a27a706-61d8-11eb-3d04-c5d7323a7249
# ╠═82cf65a0-61d8-11eb-24b3-1b1beb77e372
# ╠═9d33eee6-61d8-11eb-2a45-41a7849fc0ca
# ╠═b49bc720-61d8-11eb-2ff5-099ee4ac56c0
# ╠═75b20a8e-61d9-11eb-1bbf-6331e9accaf1
# ╠═cb50bb58-61d8-11eb-210d-0bbadabc5508
# ╠═dcac2160-61d8-11eb-1d82-b934f5d2eae7
# ╠═ef86b534-61d8-11eb-1c68-630277726bf4
# ╠═f9cadcc6-61d8-11eb-3fd1-374577903088
# ╠═41f7fcac-61dd-11eb-37b5-dd1b334525e5
# ╠═9f9ae924-61e6-11eb-2150-2ffd405f3a1e
# ╠═f7a3b9c2-61e6-11eb-2bba-f7d1d6fe1658
# ╠═045eda50-61e7-11eb-136b-47bb20bf6c6e
# ╠═2b3bcca8-61e9-11eb-204f-4d0e844680cd
# ╠═22c98e36-61e7-11eb-0389-2fcfe5f3bd11
# ╠═3c58cd92-61e9-11eb-34ec-c1b507fdffc2
# ╠═434fd42e-61e9-11eb-3178-e76dc841018f
