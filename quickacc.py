import numpy as np
import sys
import json

from collapse import metrics
from basemodel import calc_accuracy

fn="embeddings/pallet2_90_0.npz"
if len(sys.argv) > 1:
    fn = sys.argv[1]

f=np.load(fn)




