# Proof of concept mesh morphing

## Idea

We take code from a previous (propriatory) project that implements mesh morphing,
improve it along the way using skcg (also proproatory).

Some parts may be contributed to pygfx or pylinalg. The core may become
a module or example in pygfx, or grow into a separate package. We'll
see.


## Notes

(While this is still in an exploratory phase, I may add or delete stuff here.)

### Imperative vs vectorized code

The original implementation was in PScript (Python that compiled to JS) and was very imperative
in nature, because there was no Numpy we could use.

The skcg lib contains some highly vectorized code that sometimes borders
magic. It also relies on some other libraries, that we may want to avoid.

I'm trying to find a middle ground.


### Subdividing

Refactored `smooth_sphere_geometry` to use vectorized code. The
subdivision process is wrapped into a new function `subdivide_faces()`,
which we can perhaps use to increase the resolution of a morphed part
of the mesh, because you can apply the subdividing to a selection of
faces!


### Some initial benchmarks in the current state

                    MESH
                    name     sphere       knot
               nvertices     122882     120000
                  nfaces     245760     240000

                     NEW
                    init      0.440      0.457
            check_em_c_o      0.808      0.764
      check_m_and_closed      1.356      1.286
         Spit components      0.498      0.450
                  volume      0.012      0.012
              _fix_stuff      2.812      2.441
                          4188604.0  1913898.5

                    SKCG
                    init      0.002      0.004
             is_manifold      0.142      0.114
        is_manifold full     50.412     48.769
               is_closed      0.017      0.016
             is_oriented      0.022      0.020
        Split components      0.225      0.141
                  Volume      0.049      0.043
                         -4188602.41147833 -1913898.0387550928

Comments:

* The new mesh's initialization includes building a vertices2faces map.
* No big changes between the sphere and spagethi-like knot.
* The new weak manifold/closed/oriented check is about 4x slower.
* The other skcg tests are *really* fast because the `edge_index` is cached.
* The new (imperative) proper manifold check is not *that* much slower!
* The proper manifold check in skcg is reaeally slow.
* The (imperative) splitting of components is about 2x slower.
* The volume calculation is quite a bit faster than the "integral of the divergence of a function over its domain".
* The original `_fix_stuff` method was somewhat slow, we can indeed do things faster.
* The new code has some overhead to select valid faces since we allocate extra free slots.


### Optimizations and selecting algorithms

I made a matrix of the `is_xx` tests that we want, what algorithm provide
these info, and how fast these are. I used this to come up with a subset
of methods to adopt. I then tried to optimize these some more.

Methods:

* `check_edge_manifold_and_closed()` vectorized code using `np.unique`.
* `check_oriented()` similar algoritm using `np.unique`.
* `check_only_connected_by_edges()` a neat trick to check the second condition
  for manifoldness, which is relatively cheap if we split components anyway.
  The splitting of components happens by walking over the surface.

Not used/needed:

* The original (very imperatve) code that checked and repaired multiple things.
* The code that checks for face-fans at each vertex.

With this, the benchmark becomes:

                    MESH
                    name     sphere       knot
               nvertices     122882     120000
                  nfaces     245760     240000

                     NEW
                    init      0.452      0.459
     check_em_and_closed      0.160      0.155
          check_oriented      0.160      0.149
               check_fan      1.017      0.917
         Spit components      0.542      0.514
                  volume      0.012      0.012
                          4188604.0  1913898.5
