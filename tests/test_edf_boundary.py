"""Generated EDF exercises the actual MNE/loader/scan boundary, no lab data."""
import importlib.util

import mne
import numpy as np

import precompute_guard1_static_cogan as scan
from cogan_preprocessing.loader import cogan_load_raw


def test_synthetic_edf_selection_resampling_and_pre_car_scan(tmp_path):
    names=[f'A{i}' for i in range(1,9)]+['Trigger']
    rng=np.random.default_rng(33)
    data=rng.normal(0,1e-4,(9,20000))
    data[7]=0  # Deliberately dead contact.
    info=mne.create_info(names,sfreq=1000,ch_types='seeg')
    raw=mne.io.RawArray(data,info,verbose=False)
    edf=tmp_path/'synthetic.edf'
    raw.export(edf,fmt='edf',verbose=False)
    channels=tmp_path/'channels.tsv'
    channels.write_text('name\ttype\n'+''.join(
        f'{n}\t'+('TRIG' if n=='Trigger' else 'SEEG')+'\n' for n in names))
    voltage,labels,rate=cogan_load_raw(str(edf),str(channels))
    assert labels==names[:8] and rate==2048
    assert voltage.shape==(8,40960)
    assert np.max(np.abs(voltage[7]))<1e-8
    result=scan.scan_run(dict(subject_id=1001,trial_id=0,edf_path=str(edf),
                             channels_tsv_path=str(channels),power_line_hz=60))
    assert result['labels']==names[:8]
    assert result['n_elec']==8
    assert result['n_samples']==40960
    assert 'A8' in result['dead'] and 'A8' in result['static']


def test_private_training_package_is_not_a_dependency():
    assert importlib.util.find_spec('speech_decoding') is None
