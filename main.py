#!/usr/bin/env python3

"""
Estimate the position of the terrain from noisy rangefinder readings

Input is a csv file in terrain/{terrain}.csv, output is a csv file in results/{terrain}.csv
"""

import argparse
import csv
import math

from numpy.random import randn

import terrain_kf


def get_terrain(terrain, measurement_var):
    # To keep things simple use depth (NED) vs z (ENU), and assume the sensor is at the surface (0).
    # This way the rangefinder measurements and seafloor position can be compared.

    measurement_std = math.sqrt(measurement_var)
    # print(f'Measurement std: {measurement_std}')

    with open(f'terrain/{terrain}.csv', newline='') as infile:
        datareader = csv.reader(infile, delimiter=',', quotechar='|')

        # The first input row is dt, subsequent rows are ground truth
        row = next(datareader)
        dt = float(row[0])

        ts, ps, zs = [], [], []
        t = 0.0
        for row in datareader:
            p = float(row[0])

            # Time
            t += dt
            ts.append(t)

            # Ground truth
            ps.append(p)

            # Rangefinder measurement
            zs.append(p + randn() * measurement_std)

    return dt, ts, ps, zs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meas-var', type=float, default=0.01, help='measurement variance (for R), default 0.01')
    parser.add_argument('--proc-var', type=float, default=0.01, help='process variance (for Q), default 0.01')
    parser.add_argument('--terrain', type=str, default='zeros', help='load terrain from "terrain/{terrain}.csv", default "zeros"')
    args = parser.parse_args()
    print(f'Load terrain from terrain/{args.terrain}.csv, write results to results/{args.terrain}.csv')

    dt, ts, ps, zs = get_terrain(args.terrain, args.meas_var)

    kf = terrain_kf.TerrainKF(dt, args.meas_var, args.proc_var)

    with open(f'results/{args.terrain}.csv', mode='w', newline='') as outfile:

        datawriter = csv.writer(outfile, delimiter=',', quotechar='|')
        datawriter.writerow(['time', 'gt', 'rf', 'est_p', 'est_v', 'est_a', 'proj_p', 'proj_v', 'proj_a',])

        for t, p, z in zip(ts, ps, zs):
            kf.predict()
            kf.update(z)

            # Project forward by a multiple of dt and plot that as well. This simulates a large sensor delay.
            px, _ = kf.project(4)

            datawriter.writerow([t, p, z, kf.x[0], kf.x[1], kf.x[2], px[0], px[1], px[2],])
            outfile.flush()


if __name__ == '__main__':
    main()
