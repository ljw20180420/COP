#!/usr/bin/env python

import sys
from signal import signal, SIGPIPE, SIG_IGN
import numpy as np

signal(SIGPIPE, SIG_IGN)

quantile_value = float(sys.argv[1])


def output_select(pValues, lines, quantile_value, fd):
    pValues = np.array(pValues)
    pValue_thres = np.quantile(pValues, quantile_value)
    pValues[pValues < pValue_thres] = np.inf
    idx = np.argmin(pValues)
    fd.write(lines[idx])


precluster = -1
for line in sys.stdin:
    _, _, _, _, _, _, _, pValue, _, _, cluster = line.strip().split()
    pValue, cluster = float(pValue), int(cluster)
    if cluster != precluster:
        if precluster != -1:
            output_select(pValues, lines, quantile_value, sys.stdout)
        lines, pValues = [], []
    lines.append(line)
    pValues.append(pValue)
    precluster = cluster
output_select(pValues, lines, quantile_value, sys.stdout)
