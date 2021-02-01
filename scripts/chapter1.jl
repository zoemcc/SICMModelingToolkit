using Revise
using DrWatson
@quickactivate "SICMModelingToolkit"
DrWatson.greet()
using ModelingToolkit, Unitful, DifferentialEquations, Optim
using Latexify
using BenchmarkTools, LinearAlgebra, ForwardDiff, Zygote
using Quadrature, StaticArrays
using SICMModelingToolkit
using Plots
using Measures


#### section 1.4

@parameters t m k
D = Differential(t)




locals1 = generate_generic_localtuple(1)
locals2 = generate_generic_localtuple(2)
locals3 = generate_generic_localtuple(3)

velocity(locals3)

L_free_mass = L_free_particle(m)

Lagrange_equations(L_free_mass, locals3)

L_harmonic_spring_apply = L_harmonic_spring(m, k)
Lagrange_equations(L_harmonic_spring_apply, locals1)
Lagrange_equations(L_harmonic_spring_apply, locals3)



ode_generator = ode_problem_generator(L_free_particle, locals3, [m])
#Lagrangian_pre_param, locals, params = L_free_particle, locals3, [m]

#ode_func = ODEFunction(ode_generator[1])

#du = [0., 0., 0., 0., 0., 0.]
#u0 = @SVector [1., 0.5, 0.25, 0., 0., 0.]
#p0 = @SVector [1.0]
#ode_func(du, u0, p0, 0.0)

#@show du

#dyn_ode_func(du, u, p, t) = DynamicalODEFunction{true}(ode_func)



du = [0., 0., 0.]
q0 = [0., 0., 0.]
qv0 = [1.0, 0.5, 0.25]
#u0 = [1., 0.5, 0.25, 0., 0., 0.]
u0 = ArrayPartition(qv0, q0)
p0 = [1.0]
t0 = 0.0

tspan = (0.0, 2.0)

prob = ode_generator(qv0, q0, tspan, p0; saveat=0.05)


sol = solve(prob, McAte5(); dt=0.05)

plot(sol, vars=(4,5,6); dpi=350)


### kepler 

kepler_ode_generator = ode_problem_generator(L_kepler, locals2, [m, k])

begin 
    q0_kep = [0.0, 1.0]
    qv0_kep = [-1.0, 0.0]
	tspan_kep = (0.0, 5.0)
    p_kep = [3.0, 10.0]
    dt = 0.01

    kepler_prob = kepler_ode_generator(qv0_kep, q0_kep, tspan_kep, p_kep)
end

kepler_sol = solve(kepler_prob, McAte5(); dt=dt)
kepler_sol_nosymp = solve(kepler_prob, Tsit5(); dt=dt)

curxlims = (-1.1, 1.1)
curylims = (9.0/16) .* curxlims 
cur_left_margin = Measures.Length{:mm, Float64}(12)
cur_bottom_margin = Measures.Length{:mm, Float64}(7)
gr()
ellipseplot = plot(kepler_sol, vars=(4,3); dpi=500, ylims=curylims, xlims=curxlims,
    label="orbiter", xlabel="x", ylabel="y", title="Kepler Orbital Problem Symplectic", 
    left_margin=cur_left_margin, bottom_margin=cur_bottom_margin)
scatter!(ellipseplot, [0.0], [0.0]; label="planet")
save("kepler_symplectic.png", ellipseplot)
ellipseplot_nosymp = plot(kepler_sol_nosymp, vars=(4,3); dpi=500, ylims=curylims, xlims=curxlims,
    label="orbiter", xlabel="x", ylabel="y", title="Kepler Orbital Problem Nonsymplectic", 
    left_margin=cur_left_margin, bottom_margin=cur_bottom_margin)
scatter!(ellipseplot_nosymp, [0.0], [0.0]; label="planet")
save("kepler_nosymplectic.png", ellipseplot_nosymp)
plot(kepler_sol, vars=(0,3,4); dpi=350)


#### harmonic 


harmonic_ode_generator = ode_problem_generator(L_harmonic_spring, locals1, [m, k])

begin 
    q0_harm = [1.0]
    qv0_harm = [0.0]
	tspan_harm = (0.0, 5.0)
    p_harm = [1.0, 1.0]
    dt = 0.01

    harmonic_prob = harmonic_ode_generator(qv0_harm, q0_harm, tspan_harm, p_harm)
end

harmonic_sol = solve(harmonic_prob, McAte5(); dt=dt)
plot(harmonic_sol; dpi=500, left_margin=cur_left_margin, bottom_margin=cur_bottom_margin)

