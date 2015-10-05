import numpy as np
import matplotlib.pyplot as plt

from Ska.Matplotlib import plot_cxctime
from astropy.table import Table
import Ska.engarchive.fetch_eng as fetch
from Ska.engarchive.utils import logical_intervals
from kadi import events

roll_limits = Table.read('roll_limits.dat', format='ascii.fixed_width_two_line')
plot_limits = roll_limits.copy()
for i in range(len(roll_limits)-1, 0, -1):
    plot_limits.insert_row(i, [roll_limits['pitch'][i], roll_limits['rolldev'][i - 1]])

if 'dat' not in globals():
    dat = fetch.MSIDset(['roll', 'pitch'], '2015:264:03:30:00')
    dat.interpolate(dt=1.025)

if 'roll' not in globals():
    pitch = fetch.MSID('pitch', '2000:001', stat='5min')
    # pitch = pitch_all.select_intervals(events.dwells, copy=True)
    # pitch.remove_intervals(events.safe_suns)

    roll = fetch.MSID('roll', '2000:001', stat='5min')
    # roll = roll_all.select_intervals(events.dwells, copy=True)
    # roll.remove_intervals(events.safe_suns)

    # ok = logical_intervals(roll.times, np.abs(roll.midvals) < 20.1)
    # roll.select_intervals(ok)
    # pitch.select_intervals(ok)

    off_nom = logical_intervals(roll.times, np.abs(roll.midvals) > 3)
    roll_off_nom = roll.select_intervals(off_nom, copy=True)
    pitch_off_nom = pitch.select_intervals(off_nom, copy=True)

if 'greta' not in globals():
    greta = Table.read('greta_values.dat', format='ascii.basic')
    ok = (greta['s3'] == '*') & (greta['s4'] == '*')
    greta = greta[ok]


def make_ellipse():
    xe = 0.69466
    ye = 0.2588
    xmin = 0.0872
    d = 1e-5
    pitchs = []
    rolldevs = []

    for y in np.linspace(-ye + d, ye - d, 100):
        x = np.sqrt(1 - (y / ye)**2) * xe
        if x < xmin:
            continue
        z = -np.sqrt(1 - x ** 2 - y ** 2)
        pitch, rolldev = sun_body_to_pitch_rolldev(x, y, z)
        pitchs.append(pitch)
        rolldevs.append(rolldev)

    return np.array(pitchs), np.array(rolldevs)


def sun_body_to_pitch_rolldev(x, y, z):
    rolldev = np.arctan2(y, -z)
    pitch = np.arctan2(np.sqrt(y**2 + z**2), x)
    return np.degrees(pitch), np.degrees(rolldev)


def make_fish(zoom=False):
    plt.close(1)
    plt.figure(1, figsize=(6, 4))
    plt.plot(plot_limits['pitch'], plot_limits['rolldev'], '-g', lw=3)
    plt.plot(plot_limits['pitch'], -plot_limits['rolldev'], '-g', lw=3)
    plt.plot(pitch.midvals, roll.midvals, '.b', ms=1, alpha=0.7)

    p, r = make_ellipse()  # pitch, off nominal roll
    plt.plot(p, r, '-c', lw=2)

    gf = -0.08  # Fudge on pitch value for illustrative purposes
    plt.plot(greta['pitch'] + gf, -greta['roll'], '.r', ms=1, alpha=0.7)
    plt.plot(greta['pitch'][-1] + gf, -greta['roll'][-1], 'xr', ms=10, mew=2)

    if zoom:
        plt.xlim(46.3, 56.1)
        plt.ylim(4.1, 7.3)
    else:
        plt.ylim(-22, 22)
        plt.xlim(40, 180)
    plt.xlabel('Sun pitch angle (deg)')
    plt.ylabel('Sun off-nominal roll angle (deg)')
    plt.title('Mission off-nominal roll vs. pitch (5 minute samples)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('fish{}.png'.format('_zoom' if zoom else ''))


def make_fish_2():
    plt.close(1)
    plt.figure(1)
    plot_cxctime(dat.times, dat['roll'].vals)

    plt.close(2)
    plt.figure(2)
    plot_cxctime(dat.times, dat['pitch'].vals)

    plt.show()
