module SICMModelingToolkit

using ModelingToolkit, Unitful, DifferentialEquations, Optim
using RecursiveArrayTools
using SymbolicUtils
using SymbolicUtils: to_symbolic, substitute
using BenchmarkTools, LinearAlgebra, ForwardDiff, Zygote
using Quadrature, StaticArrays

include("math_base.jl")
include("lagrangians.jl")

export LocalTuple, AbstractLocalTuple, generate_generic_localtuple
export time, position, velocity, acceleration
export Lagrange_equations
export ode_problem_generator
export L_free_particle, L_harmonic_spring, L_kepler

end
