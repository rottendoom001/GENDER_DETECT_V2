# -*- coding: utf-8 -*-
"""
test_gender_detector
----------------------------------
Tests for `gender_detector` module.
"""
import numpy as np
import scipy.io.wavfile
from gender_detector import GenderDetector
from io import BytesIO

def test_hps():
    """
    Test the hps function.
    """
    arr = [255, 255, 200, 300, 500, 100, 600, 0]
    expected = [
        16581375.0,
        15300000.0,
        60000000.0,
        180000.0,
        500.0,
        100.0,
        600.0,
        0.0]
    result = GenderDetector().hps(arr)
    result = [float(item) for item in result]
    assert np.allclose(expected, result)


def test_calculate_spectrum_cepstrum_mono():
    """
    Test the calculate_spectrum_cepstrum function
    when the signal is mono.
    """
    arr = [255, 255, 200, 300, 500, 100, 600, 0]
    arr = np.array(arr)
    fs = 44100

    expected_y = np.array(
        [10793861000.0, 18623418.961417913,
         71.06335201775947, 735.660880806231]
    )
    expected_freq = np.array([0., 5512.5, 11025., 16537.5])
    expected_ceps = np.array(
        [6.11016956, 0.85929401,
         -0.12800978, 0.85929401]
    )

    y, freq, ceps = GenderDetector().calculate_spectrum_cepstrum(arr, fs)
    assert np.allclose(
        expected_y,
        y) and np.allclose(
        expected_freq,
        freq) and np.allclose(
            expected_ceps,
        ceps)


def test_calculate_spectrum_cepstrum_stereo():
    """
    Test the calculate_spectrum_cepstrum function
    when the signal is stereo.
    """
    arr = [
        [255, 0],
        [255, 0],
        [200, 0],
        [300, 0],
        [500, 0],
        [100, 0],
        [600, 0],
        [0, 0]
    ]
    arr = np.array(arr)
    print(arr)
    fs = 44100

    expected_y = np.array(
        [10793861000.0, 18623418.961417913,
         71.06335201775947, 735.660880806231])
    expected_freq = np.array([0., 5512.5, 11025., 16537.5])
    expected_ceps = np.array([6.11016956, 0.85929401,
                              -0.12800978, 0.85929401])

    y, freq, ceps = GenderDetector().calculate_spectrum_cepstrum(arr, fs)
    assert np.allclose(
        expected_y,
        y) and np.allclose(
        expected_freq,
        freq) and np.allclose(
            expected_ceps,
        ceps)


def test_calculate_modulation_index():
    """
    Test the calculate_modulation_index function.
    """
    arr = [255, 255, 200, 300, 500, 100, 600, 0]
    arr = np.array(arr)
    mindom = 11025.
    maxdom = 5512.5
    dfrange = 1000.0
    expected = 0.265
    result = GenderDetector().calculate_modulation_index(
        arr, mindom, maxdom, dfrange
    )
    assert np.allclose(expected, result)


def test_calculate_modulation_index_zero():
    """
    Test the calculate_modulation_index function when
    mindom == maxdom.
    """
    arr = [255, 255, 200, 300, 500, 100, 600, 0]
    arr = np.array(arr)
    mindom = 5512.5
    maxdom = 5512.5
    dfrange = 1000.0
    expected = 0.
    result = GenderDetector().calculate_modulation_index(
        arr, mindom, maxdom, dfrange
    )
    assert np.allclose(expected, result)


def test_get_n_fundamental_frequencies():
    """
    Test the get_n_fundamental_frequencies function.
    """
    arr = [
        (1433045920722.5535,
         1001.4545454545454),
        (1432482084982.5637,
         546.32727272727266),
        (1429332907671.3792,
         947.4909090909091),
        (1424138212407.696,
         876.07272727272721),
        (1423571652995.4846,
         977.74545454545455),
        (1420207470649.9392,
         932.94545454545448),
        (1417896507230.5828,
         990.39999999999998),
        (1413376292432.0315,
         1010.3272727272727),
        (1408203258629.1887,
         1017.8909090909091),
        (1402265894500.063,
         1049.7454545454545)]
    n = 3
    expected = np.array(
        [1001.4545454545454, 546.32727272727266, 947.4909090909091])
    result = GenderDetector().get_n_fundamental_frequencies(n, arr)
    assert np.allclose(expected, result)


def test_calculate_all_spec_props():
    """
    Test the calculate_all_spec_props function.
    """
    expected = [8.26875000e+03, 6.16316236e+03,
                8.26875000e+03, 4.13437500e+03,
                1.24031250e+04, 8.26875000e+03,
                0.00000000e+00, -1.36000000e+00,
                1.01140426e+00, 1.18447081e-04,
                1.00145455e+03, 0.00000000e+00,
                1.65375000e+04, 1.65375000e+04,
                3.33333333e-01, 1.92518695e+00,
                6.11016956e+00, -1.28009780e-01,
                1.00145455e+03, 5.46327273e+02,
                9.47490909e+02, 8.76072727e+02,
                9.77745455e+02, 9.32945455e+02,
                9.90400000e+02, 1.01032727e+03,
                1.01789091e+03, 1.04974545e+03]

    y = np.array(
        [10793861000.0, 18623418.961417913,
         71.06335201775947, 735.660880806231]
    )
    frq = np.array([0., 5512.5, 11025., 16537.5])
    ceps = np.array([6.11016956, 0.85929401, -0.12800978, 0.85929401])
    esp_frecuencia_pairs = [
        (1433045920722.5535,
         1001.4545454545454),
        (1432482084982.5637,
         546.32727272727266),
        (1429332907671.3792,
         947.4909090909091),
        (1424138212407.696,
         876.07272727272721),
        (1423571652995.4846,
         977.74545454545455),
        (1420207470649.9392,
         932.94545454545448),
        (1417896507230.5828,
         990.39999999999998),
        (1413376292432.0315,
         1010.3272727272727),
        (1408203258629.1887,
         1017.8909090909091),
        (1402265894500.063,
         1049.7454545454545)]
    result = GenderDetector().calculate_all_spec_props(
        frq, y, ceps, esp_frecuencia_pairs
    )
    assert np.allclose(expected, result)


def test_preprocess():
    """
    Test the preprocess function.
    """
    inputFileName = './tests/wav/1F.wav'
    fs, signal = scipy.io.wavfile.read(inputFileName)
    expected = [
        5.87534884e+02, 2.95871868e+02, 5.87534884e+02, 3.31348837e+02,
        8.43720930e+02, 5.12372093e+02, -1.17632943e-15, -1.20000008e+00,
        8.47444655e+00, 2.75899724e-02, 2.07813953e+02, 7.51627907e+01,
        1.09990698e+03, 1.02474419e+03, 1.81554103e-04, 7.07795704e-04,
        1.01110553e+01, -2.22503906e-01, 2.07813953e+02, 2.12465116e+02,
        2.23627907e+02, 2.37023256e+02, 2.35906977e+02, 2.34418605e+02,
        2.06139535e+02, 2.36093023e+02, 2.36465116e+02, 2.23813953e+02
    ]
    result = GenderDetector().preprocess(fs, signal)
    assert np.allclose(expected, result)


def test_process():
    """
    Test the process function.
    """
    response = {
        "data": {
            "gender": {
                "id": "FEMALE",
                "name": "The gender of this person is : FEMALE"
            }
        }
    }

    inputFileName = './tests/wav/1F.wav'
    fs, signal = scipy.io.wavfile.read(inputFileName)
    result = GenderDetector().process(fs, signal)
    assert (
        response["data"]["gender"]["id"] == result["data"]["gender"]["id"] and
        response["data"]["gender"]["name"] == result["data"]["gender"]["name"]
    )

'''
fs, signal = scipy.io.wavfile.read('./tests/wav/1F.wav')
print ("fs:", fs)
for value in range(0, 40):
    print(signal[value])

print ("/////////////////")

f = open('./tests/wav/1F.bin', 'rb')
bynary_data = f.read()

f2 = BytesIO(bynary_data)
#fs, signal = process_wav_binary(f2)
fs, signal = scipy.io.wavfile.read(f2)
print ("fs:", fs)
for value in range(0, 40):
    print(signal[value])
f.close()
'''
