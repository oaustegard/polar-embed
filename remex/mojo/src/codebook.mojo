"""Lloyd-Max scalar quantizer codebook for N(0, 1/d).

Mirrors `remex.codebook.lloyd_max_codebook`. The Mojo version uses
`+/-INF_SENTINEL = ±50` for the outer interval edges — far enough into
the tails that cdf(±50*sigma) is 1.0/0.0 in float64.
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from src.mathx import normal_cdf, normal_pdf


comptime N_ITER_DEFAULT = 300
comptime INF_SENTINEL = 50.0  # any value where cdf(x*sigma) saturates


struct Codebook(Movable):
    """Holds a Lloyd-Max codebook for a fixed (d, bits)."""
    var boundaries: UnsafePointer[Float32, MutExternalOrigin]   # length n_levels - 1
    var centroids: UnsafePointer[Float32, MutExternalOrigin]    # length n_levels
    var n_levels: Int
    var bits: Int

    def __init__(out self, bits: Int):
        self.bits = bits
        self.n_levels = 1 << bits
        # Allocate centroids first; boundaries gets at least one slot.
        self.centroids = alloc[Float32](self.n_levels)
        var n_b = self.n_levels - 1
        if n_b < 1:
            n_b = 1
        self.boundaries = alloc[Float32](n_b)

    def get_centroid(self, j: Int) -> Float32:
        return self.centroids[j]

    def get_boundary(self, j: Int) -> Float32:
        return self.boundaries[j]

    def __del__(deinit self):
        self.boundaries.free()
        self.centroids.free()


def lloyd_max_codebook(d: Int, bits: Int, n_iter: Int = N_ITER_DEFAULT) -> Codebook:
    """Build a Lloyd-Max codebook for N(0, 1/d) coordinates."""
    var n_levels = 1 << bits
    var sigma = 1.0 / sqrt(Float64(d))
    var sigma2 = sigma * sigma

    # Float64 working buffers for numerical stability
    var c = alloc[Float64](n_levels)
    var b_lo = alloc[Float64](n_levels)
    var b_hi = alloc[Float64](n_levels)

    # Initialize centroids: linspace(-3*sigma, 3*sigma, n_levels)
    if n_levels == 1:
        c[0] = 0.0
    else:
        var lo = -3.0 * sigma
        var hi = 3.0 * sigma
        var step = (hi - lo) / Float64(n_levels - 1)
        for i in range(n_levels):
            c[i] = lo + Float64(i) * step

    for _ in range(n_iter):
        # Compute bounds
        b_lo[0] = -INF_SENTINEL
        b_hi[n_levels - 1] = INF_SENTINEL
        for k in range(n_levels - 1):
            var mid = Float64(0.5) * (c[k] + c[k + 1])
            b_hi[k] = mid
            b_lo[k + 1] = mid

        # Update centroids: conditional mean on each interval
        for j in range(n_levels):
            var lo_ = b_lo[j]
            var hi_ = b_hi[j]
            var ca = normal_cdf(lo_, sigma)
            var cdfb = normal_cdf(hi_, sigma)
            var prob = cdfb - ca
            if prob > Float64(1e-15):
                var pa = normal_pdf(lo_, sigma)
                var pb = normal_pdf(hi_, sigma)
                var newc = sigma2 * (pa - pb) / prob
                c[j] = newc

    var cb = Codebook(bits)
    for j in range(n_levels):
        cb.centroids[j] = Float32(c[j])
    for j in range(n_levels - 1):
        var mid = Float64(0.5) * (c[j] + c[j + 1])
        cb.boundaries[j] = Float32(mid)

    c.free()
    b_lo.free()
    b_hi.free()
    return cb^
