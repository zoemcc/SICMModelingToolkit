function L_free_particle(mass) 
    function L_apply(locals::AbstractLocalTuple)
        v = velocity(locals)
        1//2 * mass * rdot(v,v)
    end
end


function L_harmonic_spring(mass, spring) 
    function L_apply(locals::AbstractLocalTuple)
        q = position(locals)
        qv = velocity(locals)
        1//2 * mass * rdot(qv, qv) - 1//2 * spring * rdot(q, q) 
    end
end

function L_kepler(particlemass, planetmass)
    function L_apply(locals::AbstractLocalTuple)
        q = position(locals)
        qv = velocity(locals)
        1//2 * particlemass * rdot(qv, qv) + planetmass / sqrt(rdot(q, q))
    end
end
