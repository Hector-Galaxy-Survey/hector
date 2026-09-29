"""Tests for the secondary-standard quality metrics in dr/ss_quality.py.

Synthetic Moffat stars are placed on the real layouts of the 37-core SS
hexabundles (H and U) from utils/Fibre_slitInfo_*.csv.
"""

import os

import numpy as np
import pandas as pd
import pytest

import hector
from hector.config import plate_scale, fibre_diameter_arcsec
from hector.dr import ss_quality as ssq

FIBRE_FILE = os.path.join(
    hector.__path__[0],
    'utils/Fibre_slitInfo_final_updated03Mar2022for_swapped_fibres_BrokenHexaM.csv')


def bundle_layout(name, include_broken=False):
    """Fibre centres of a hexabundle in arcsec, relative to the bundle centre."""
    table = pd.read_csv(FIBRE_FILE)
    rows = table[table['Bundle/plate'] == name]
    kind = rows['Hexabundle or sky fibre']
    keep = (kind == 'H') | (include_broken & (kind == 'BROKEN'))
    rows = rows[keep]
    # plate_scale is arcsec/mm and positions are in microns
    x = rows['Bundle_Xc'].to_numpy(float) * plate_scale / 1000.0
    y = rows['Bundle_Yc'].to_numpy(float) * plate_scale / 1000.0
    return x - x.mean(), y - y.mean()


def moffat_fibre_flux(x0, y0, xf, yf, fwhm=2.0, beta=4.0, total=1e4,
                      n_sub=15):
    """Moffat flux integrated over each fibre core (sub-sampled disc)."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    r_core = fibre_diameter_arcsec / 2.0
    g = np.linspace(-r_core, r_core, n_sub)
    gx, gy = np.meshgrid(g, g)
    in_core = gx ** 2 + gy ** 2 <= r_core ** 2
    dx, dy = gx[in_core], gy[in_core]
    pix_area = (g[1] - g[0]) ** 2
    norm = (beta - 1.0) / (np.pi * alpha ** 2)
    flux = np.empty(len(xf))
    for i, (xc, yc) in enumerate(zip(xf, yf)):
        r2 = (xc + dx - x0) ** 2 + (yc + dy - y0) ** 2
        flux[i] = total * np.sum(norm * (1 + r2 / alpha ** 2) ** -beta) * pix_area
    return flux


def chunked(xf, yf, flux):
    return {'xfibre': xf, 'yfibre': yf, 'data': flux[:, None]}


@pytest.fixture(scope='module')
def layout_u():
    return bundle_layout('U')


def directions(xf, yf):
    """Unit-circle angles (rad) towards a corner and the adjacent side."""
    phi_corner = ssq.hexagon_orientation(xf, yf)
    return phi_corner, phi_corner + np.radians(30.0)


def outermost_fibre_towards(phi, xf, yf):
    """Position of the outer-ring fibre closest in angle to phi."""
    outer = np.where(ssq.outer_fibres(xf, yf))[0]
    dphi = np.angle(np.exp(1j * (np.arctan2(yf[outer], xf[outer]) - phi)))
    i = outer[np.argmin(np.abs(dphi))]
    return xf[i], yf[i]


def test_layout_is_37_core_hexagon(layout_u):
    xf, yf = layout_u
    assert len(xf) == 37
    # 1 + 6 + 12 + 18: the outermost ring has 18 fibres
    outer = ssq.outer_fibres(xf, yf)
    assert outer.sum() == 18
    # The six outer fibres furthest out are the corners, 60 deg apart and
    # aligned with the fitted orientation
    r = np.hypot(xf, yf)
    corners = np.argsort(np.where(outer, r, 0))[-6:]
    phi = np.arctan2(yf[corners], xf[corners])
    for p in phi:
        assert ssq.corner_angle(np.cos(p), np.sin(p), xf, yf) < 3.0


def test_centred_star(layout_u):
    xf, yf = layout_u
    vertices, _ = ssq.footprint_hull(xf, yf)
    apothem = ssq.signed_distance_to_polygon(0.0, 0.0, vertices)
    flux = moffat_fibre_flux(0.0, 0.0, xf, yf)
    keys = dict((k, v) for k, v, _ in ssq.assess_secondary_quality(
        chunked(xf, yf, flux), {'xcen_ref': 0.0, 'ycen_ref': 0.0}, fwhm=2.0))
    assert keys['SSEDGE'] == pytest.approx(apothem + fibre_diameter_arcsec / 2)
    assert keys['SSEDGEF'] == pytest.approx(keys['SSEDGE'] / 2.0)
    assert keys['SSOUTER'] < 0.1
    assert keys['SSASYM'] < 0.05


def test_edge_distance_depends_on_azimuth(layout_u):
    """At the same radius, a star towards a corner is further from the edge
    than one towards the middle of a flat side."""
    xf, yf = layout_u
    phi_corner, phi_flat = directions(xf, yf)
    r = 3.5
    xc, yc = r * np.cos(phi_corner), r * np.sin(phi_corner)
    xe, ye = r * np.cos(phi_flat), r * np.sin(phi_flat)

    d_corner = ssq.edge_distance(xc, yc, xf, yf)
    d_flat = ssq.edge_distance(xe, ye, xf, yf)
    assert d_corner > d_flat + 0.3
    assert ssq.corner_angle(xc, yc, xf, yf) == pytest.approx(0.0, abs=2.0)
    assert ssq.corner_angle(xe, ye, xf, yf) == pytest.approx(30.0, abs=2.0)


def test_edge_star_is_truncated(layout_u):
    """A star at the flat edge has much of its observed light in the outer
    ring, and the observed centroid is pulled inwards of the true one."""
    xf, yf = layout_u
    _, phi_flat = directions(xf, yf)
    # centre the star on the outer fibre in the middle of a side
    x0, y0 = outermost_fibre_towards(phi_flat, xf, yf)
    flux = moffat_fibre_flux(x0, y0, xf, yf)
    keys = dict((k, v) for k, v, _ in ssq.assess_secondary_quality(
        chunked(xf, yf, flux), {'xcen_ref': x0, 'ycen_ref': y0}, fwhm=2.0))
    assert keys['SSEDGE'] == pytest.approx(fibre_diameter_arcsec / 2,
                                           abs=0.1)
    # A side has 4 fibres (corner, 2 middle, corner), so the middle fibres
    # sit ~20 deg from the nearest corner rather than at 30
    assert 15.0 < keys['SSPHI'] < 25.0
    assert keys['SSOUTER'] > 0.5
    assert keys['SSASYMR'] > 0.2


def test_star_outside_bundle_is_negative(layout_u):
    xf, yf = layout_u
    assert ssq.edge_distance(10.0, 0.0, xf, yf) < 0


def test_missing_outer_fibre_moves_edge_in(layout_u):
    """Dropping the corner fibre (as for a broken fibre) brings the edge in."""
    xf, yf = layout_u
    phi_corner, _ = directions(xf, yf)
    cx, cy = outermost_fibre_towards(phi_corner, xf, yf)
    keep = ~((xf == cx) & (yf == cy))
    x0, y0 = 0.8 * cx, 0.8 * cy
    assert (ssq.edge_distance(x0, y0, xf[keep], yf[keep]) <
            ssq.edge_distance(x0, y0, xf, yf) - 0.3)


def test_bundle_h_with_broken_fibre():
    """H has a broken fibre in the layout file; metrics still work."""
    xf, yf = bundle_layout('H')
    assert len(xf) == 36
    flux = moffat_fibre_flux(0.0, 0.0, xf, yf)
    keys = dict((k, v) for k, v, _ in ssq.assess_secondary_quality(
        chunked(xf, yf, flux), {'xcen_ref': 0.0, 'ycen_ref': 0.0}, fwhm=2.0))
    assert np.all(np.isfinite([keys[k] for k in ('SSEDGE', 'SSOUTER',
                                                 'SSASYM')]))


def test_fit_model_flux_return_chi2(layout_u):
    """return_chi2 adds a chi^2 without changing the default return value."""
    from hector.dr.fluxcal2 import fit_model_flux
    xf, yf = layout_u
    wl = np.linspace(4000, 7000, 6)
    rng = np.random.default_rng(1)
    model = np.column_stack([moffat_fibre_flux(1.0, -0.5, xf, yf)
                             for _ in wl])
    var = np.abs(model) + 5
    data = model + rng.normal(0, np.sqrt(var))
    name = 'ref_centre_alpha_circ_hdratm'
    fixed = {'zenith_direction': 0.5, 'zenith_distance': 0.0,
             'temperature': 10.0, 'pressure': 700.0, 'vapour_pressure': 8.0}
    params = fit_model_flux(data, var, xf, yf, wl, name,
                            fixed_parameters=fixed)
    assert isinstance(params, dict)
    assert params['xcen_ref'] == pytest.approx(1.0, abs=0.1)
    params2, chi2 = fit_model_flux(data, var, xf, yf, wl, name,
                                   fixed_parameters=fixed, return_chi2=True)
    assert params2['xcen_ref'] == pytest.approx(params['xcen_ref'])
    assert 0.5 < chi2 < 3.0


def test_extracted_to_summed():
    data = np.ones((4, 10))
    good = np.array([True, True, True, False])
    assert ssq.extracted_to_summed(np.full(10, 6.0), data, good) == 2.0
