#!/usr/bin/env python

import resource

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
# limit memory usage to 50 GB
resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, hard))

import numpy as np
from sklearn.svm import LinearSVC

train_eval_ids = np.random.randint(0, 26, (334800, 5503))
train_eval_bind = np.random.randint(0, 2, (334800,))

classifier = LinearSVC(max_iter=1000, random_state=63036)
classifier.fit(train_eval_ids, train_eval_bind)
