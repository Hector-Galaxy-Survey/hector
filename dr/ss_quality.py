"""
Quality metrics for secondary standard (SS) star extractions.

The SS flux is extracted by fitting a PSF model to the fibres of a small
hexabundle (37 cores for H and U) and integrating the model to get the total
flux.  When the star sits near the edge of the bundle, or the S/N is low, the
total flux is poorly constrained.  The functions here quantify that, so that
unreliable SS calibrations can be flagged.

The metrics deliberately avoid a model-based "enclosed light fraction": the
extraction already relies on the same PSF model to extrapolate the total flux,
so using it to judge the extrapolation would be circular.  Instead:

SSEDGE   signed distance (arcsec) from the fitted centroid to the edge of the
         bundle footprint, positive inside.  The footprint is the convex hull
         of the good-fibre centres grown by the core radius, so it follows the
         hexagonal shape, the bundle size and any dead fibres.
SSEDGEF  SSEDGE in units of the fitted FWHM.
SSPHI    angle (deg) between the centroid direction and the nearest corner
         of the hexabundle, as seen from the footprint centre:
         0 = towards a corner, 30 = towards the middle of a side.
SSOUTER  fraction of the observed (summed, model-free) flux that falls in the
         outermost fibres, i.e. those on the footprint boundary.
SSASYM   offset (arcsec) between the flux-weighted centroid of the observed
         light and the fitted centroid.  SSASYMR is its radial component,
         positive when the fit places the star further out than the observed
         light (i.e. the fit is extrapolating beyond the bundle).

These are all diagnostics; thresholds are set separately.
"""

import numpy as np
from scipy.spatial import ConvexHull

from ..config import fibre_diameter_arcsec

CORE_RADIUS = fibre_diameter_arcsec / 2.0


def footprint_hull(xfibre, yfibre):
    """Return the vertices (anticlockwise, closed) of the convex hull of the
    fibre centres, and the centroid of those centres."""
    points = np.column_stack((xfibre, yfibre))
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    centre = np.array([np.mean(xfibre), np.mean(yfibre)])
    return vertices, centre


def signed_distance_to_polygon(x, y, vertices):
    """Signed distance from (x, y) to the boundary of a convex polygon.

    Positive inside, negative outside.  vertices are ordered anticlockwise
    (as returned by ConvexHull in 2D) and not repeated at the end."""
    p = np.array([x, y], dtype=float)
    a = vertices
    b = np.roll(vertices, -1, axis=0)
    ab = b - a
    ap = p - a
    t = np.clip(np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0)
    closest = a + t[:, None] * ab
    dist = np.min(np.hypot(*(p - closest).T))
    # Inside a convex anticlockwise polygon, p is to the left of every edge
    cross = ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]
    inside = np.all(cross >= 0.0)
    return dist if inside else -dist


def edge_distance(xcen, ycen, xfibre, yfibre, core_radius=CORE_RADIUS):
    """Signed distance (arcsec) from (xcen, ycen) to the bundle footprint edge.

    The footprint is the convex hull of the fibre centres grown by the core
    radius (a Minkowski sum with a disc), so the distance is simply the
    distance to the hull plus the core radius."""
    vertices, _ = footprint_hull(xfibre, yfibre)
    return signed_distance_to_polygon(xcen, ycen, vertices) + core_radius


def fibre_pitch(xfibre, yfibre):
    """Median nearest-neighbour spacing of the fibre centres."""
    d = np.hypot(xfibre[:, None] - xfibre[None, :],
                 yfibre[:, None] - yfibre[None, :])
    np.fill_diagonal(d, np.inf)
    return np.median(np.min(d, axis=1))


def hexagon_orientation(xfibre, yfibre):
    """Position angle (rad) of a corner of the hexabundle.

    Real bundles are not perfect hexagons (the sides bow outwards slightly
    and fibres can be missing), so the convex hull has more than six
    vertices.  Instead use the six-fold harmonic of the outer fibres, which
    peaks towards the corners where the outer fibres lie furthest out."""
    _, centre = footprint_hull(xfibre, yfibre)
    outer = outer_fibres(xfibre, yfibre)
    dx, dy = xfibre[outer] - centre[0], yfibre[outer] - centre[1]
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    harmonic = np.sum((r - np.mean(r)) * np.exp(6j * phi))
    return np.angle(harmonic) / 6.0


def corner_angle(xcen, ycen, xfibre, yfibre):
    """Angle (deg) between the direction to (xcen, ycen) and the nearest
    corner of the hexabundle, measured from the footprint centre.
    0 = towards a corner, 30 = towards the middle of a side."""
    _, centre = footprint_hull(xfibre, yfibre)
    dx, dy = xcen - centre[0], ycen - centre[1]
    if np.hypot(dx, dy) == 0.0:
        return np.nan
    phi = np.degrees(np.arctan2(dy, dx) -
                     hexagon_orientation(xfibre, yfibre))
    return np.abs((phi + 30.0) % 60.0 - 30.0)


def outer_fibres(xfibre, yfibre, tol=None):
    """Boolean mask of fibres lying on the footprint boundary.

    ConvexHull drops collinear points, so fibres along a flat edge are found
    by their distance to the hull rather than from hull.vertices.  The next
    ring in lies ~0.87 pitch inside the hull, so half a pitch separates the
    two cleanly."""
    vertices, _ = footprint_hull(xfibre, yfibre)
    if tol is None:
        tol = 0.5 * fibre_pitch(xfibre, yfibre)
    d = np.array([signed_distance_to_polygon(x, y, vertices)
                  for x, y in zip(xfibre, yfibre)])
    return d < tol


def outer_flux_fraction(fibre_flux, xfibre, yfibre):
    """Fraction of the summed observed flux in the outermost fibres."""
    good = np.isfinite(fibre_flux)
    total = np.sum(fibre_flux[good])
    if not total > 0:
        return np.nan
    outer = outer_fibres(xfibre, yfibre)
    return np.sum(fibre_flux[good & outer]) / total


def observed_centroid(fibre_flux, xfibre, yfibre):
    """Flux-weighted centroid of the observed light (negatives clipped)."""
    w = np.where(np.isfinite(fibre_flux), np.clip(fibre_flux, 0.0, None), 0.0)
    if not np.sum(w) > 0:
        return np.nan, np.nan
    return np.sum(w * xfibre) / np.sum(w), np.sum(w * yfibre) / np.sum(w)


def centroid_offset(fibre_flux, xfibre, yfibre, xcen, ycen):
    """Offset between the observed and fitted centroids.

    Returns (total offset, radial component).  The radial component is
    measured along the direction from the footprint centre to the fitted
    centroid, and is positive when the fitted centroid lies further out
    than the observed one."""
    xo, yo = observed_centroid(fibre_flux, xfibre, yfibre)
    _, centre = footprint_hull(xfibre, yfibre)
    dx, dy = xcen - xo, ycen - yo
    total = np.hypot(dx, dy)
    rx, ry = xcen - centre[0], ycen - centre[1]
    r = np.hypot(rx, ry)
    radial = (dx * rx + dy * ry) / r if r > 0 else 0.0
    return total, radial


def fibre_flux_from_chunks(chunked_data):
    """Summed flux per fibre from read_chunked_data output."""
    return np.nansum(chunked_data['data'], axis=1)


def assess_secondary_quality(chunked_data, psf_parameters, fwhm=None):
    """Compute the SS quality metrics for one extraction.

    chunked_data   output of fluxcal2.read_chunked_data (good fibres only)
    psf_parameters fitted PSF parameters, containing xcen_ref and ycen_ref
    fwhm           fitted FWHM in arcsec, used to scale SSEDGE

    Returns an ordered list of (header key, value, comment) tuples."""
    xf = np.asarray(chunked_data['xfibre'], dtype=float)
    yf = np.asarray(chunked_data['yfibre'], dtype=float)
    xcen = float(psf_parameters['xcen_ref'])
    ycen = float(psf_parameters['ycen_ref'])
    fibre_flux = fibre_flux_from_chunks(chunked_data)

    edge = edge_distance(xcen, ycen, xf, yf)
    edge_fwhm = edge / fwhm if fwhm else np.nan
    phi = corner_angle(xcen, ycen, xf, yf)
    outer = outer_flux_fraction(fibre_flux, xf, yf)
    asym, asym_r = centroid_offset(fibre_flux, xf, yf, xcen, ycen)

    return [
        ('SSEDGE', edge, 'SS centroid to bundle edge (arcsec, +ve inside)'),
        ('SSEDGEF', edge_fwhm, 'SSEDGE in units of PSF FWHM'),
        ('SSPHI', phi, 'SS angle from nearest bundle corner (deg)'),
        ('SSOUTER', outer, 'Frac of observed SS flux in outer fibres'),
        ('SSASYM', asym, 'Observed-fitted SS centroid offset (arcsec)'),
        ('SSASYMR', asym_r, 'Radial part of SSASYM, +ve fit further out'),
    ]


def extracted_to_summed(observed_flux, ifu_data, good_fibre):
    """Median ratio of the model-extracted flux to the flux summed over the
    good fibres of the bundle.  Records how much model correction was
    applied; it does not say whether that correction is right."""
    summed = np.nansum(ifu_data[good_fibre, :], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = observed_flux / summed
    ratio = ratio[np.isfinite(ratio)]
    return np.median(ratio) if ratio.size else np.nan
