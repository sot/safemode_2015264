import matplotlib.pyplot as plt
import Ska.engarchive.fetch_eng as fetch
from Ska.Matplotlib import plot_cxctime
from kadi import events


def plot_fss(start='2015:240', stop='2015:265'):
    dat = fetch.MSIDset(['roll', 'roll_fss', 'pitch', 'pitch_fss'], start, stop)
    dat.interpolate(2.05, bad_union=True)
    #for msid in dat.keys():
    #    dat[msid].select_intervals(events.dwells)

    plt.close(1)
    plt.figure(1, figsize=(6, 4))
    plt.subplot(2, 1, 1)
    plot_cxctime(dat['roll'].times, dat['roll_fss'].vals - dat['roll'].vals, '.', ms=1)
    plt.ylabel('Delta angle (deg)')
    plt.grid()
    plt.title('Roll (FSS = SPM) - Roll (attitude, ephem = MCC)')

    plt.subplot(2, 1, 2)
    plot_cxctime(dat['pitch'].times, dat['pitch_fss'].vals - dat['pitch'].vals, '.', ms=1)
    plt.ylabel('Delta angle (deg)')
    plt.title('Pitch (FSS = SPM) - Pitch (attitude, ephem = MCC)')
    plt.grid()
    plt.tight_layout()

    plt.savefig('fss_pitch_roll_{}_{}.png'.format(start, stop))
    return dat
