"""
Creates directories for test data and results.
"""

import shutil
import os


if os.path.exists('./test_panels'):
    if os.path.isdir('./test_panels'):
        shutil.rmtree('./test_panels')
    else:
        os.remove('./test_panels')
os.makedirs('./test_panels')

if os.path.exists('./test_results'):
    if os.path.isdir('./test_results'):
        shutil.rmtree('./test_results')
    else:
        os.remove('./test_results')
os.makedirs('./test_results')

if os.path.exists('./test'):
    if os.path.isdir('./test'):
        shutil.rmtree('./test')
    else:
        os.remove('./test')
os.makedirs('./test')
