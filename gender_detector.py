import numpy as np
import pickle
from scipy.stats import kurtosis, skew, entropy, gmean
from datetime import date


class GenderDetector(object):
    """
    Identify the gender through voice processing.
    """

    def __init__(self):
        # CARGAMOS EL MODELO
        self.model = pickle.load(open("./model/xgboost_model.dat", "rb"))
        self.mono = 1
        self.q75 = 75
        self.q25 = 25
        self.min_frq = 75
        self.max_frq = 1100
        self.c = 10

    def hps(self, arr):
        """
        Calculate the hps algorithm.

        :param arr: Array that contains the audio frequencies after applying
        the fft
        :return: Array that contains the fundamental frequencies of the audio
        signalstring with language and confidence interval
        """
        r = arr
        d2 = []
        d3 = []
        i = 0
        # Diezmar en 2
        for v in arr:
            if i % 2 == 0:
                d2.append(v)
            i += 1
        # Diezmar en 3
        i = 0
        for v in arr:
            if i % 3 == 0:
                d3.append(v)
            i += 1
        d2 = np.array(d2)
        d3 = np.array(d3)
        # Multiplicar por d2
        i = 0
        for v in d2:
            r[i] = r[i] * v
            i += 1
        # Multiplicar por d3
        i = 0
        for v in d3:
            r[i] = r[i] * v
            i += 1
        return r

    def calculate_spectrum_cepstrum(self, y, fs):
        """
        Calculates frequency spectrum and cepstrum.

        :param y: Array that contains the samples in the time domain
        of the audio signal
        :param fs: Integer that contains the sampling frequency
        :return y: Array that contains the espectrum of the audio signal
        :return freq: Array that contains the frequencies of the audio signal
        :return ceps: Array that contains cepstrum of the audio signal
        """
        # Hay algunos audios que son estereo, se toma un lado
        y = y[:, 0] if y.ndim > self.mono else y
        n = len(y)
        j = int(n / 2)
        y = np.fft.fft(y)
        freq = np.fft.fftfreq(len(y), 1.0 / fs)
        freq = freq[range(j)]
        y = y[range(j)]
        y = abs(y)
        ceps = np.fft.ifft(np.log(y)).real
        y = self.hps(y)
        return y, freq, ceps

    def calculate_modulation_index(self, y, mindom, maxdom, dfrange):
        """
        Calculates the modulation index of frequency spectrum.

        :param y: Array that contains the samples in the time domain
        of the audio signal
        :param mindom: Float with minimum of dominant frequency measured across
        acoustic signal
        :param maxdom: Float with maximum of dominant frequency measured across
        acoustic signal
        :param dfrange: Float with range of dominant frequency measured across acoustic signal
        :return modindx: Float with modulation index value
        """
        changes = []
        for j in range(len(y) - 1):
            change = abs(y[j] - y[j + 1])
            changes.append(change)
        modindx = 0 if(mindom == maxdom) else np.mean(changes) / dfrange
        return modindx

    def get_n_fundamental_frequencies(self, n, arr):
        """
        Calculates the modulation index of frequency spectrum.

        :param y: Array that contains the frequencies of the audio signal
        :param mindom: Float with minimum of dominant frequencymeasured across
        acoustic signal
        :param maxdom: Float with maximum of dominant frequencymeasured across
        acoustic signal
        :param dfrange: Float with range of dominant frequency measured across acoustic signal
        :return modindx: Float with modulation index value
        """
        fundamental = []
        i = 1
        for v in arr:
            fundamental.append(float(v[1]))
            if n == i:
                break
            i += 1
        return fundamental

    def calculate_all_spec_props(self, frq, Y, ceps, esp_frecuencia_pairs):
        """
        Calculates all the acoustic properties including
        the fundamental frequencies.

        :param Y: Array that contains the power spectrum (decibels)
        :param frq: Array that contains the frequencies of the audio signal
        :param ceps: Array that contains cepstrum of the audio signal
        :param esp_frecuencia_pairs: lista that contains tuples of the form:
        (decibels (axis Y), frecuency(axis X))

        :return : Array with the follow structure:
        mean frequency (in kHz),
        standard deviation of frequency,
        median frequency,
        first quantile,
        third quantile,
        interquantile range,
        skewness,
        kurtosis,
        spectral entropy,
        spectral flatness,
        peak frequency,
        average of fundamental frequency (espectrum),
        minimum fundamental frequency (espectrum),
        maximum fundamental frequency (espectrum),
        average of dominant frequency (cepstrum),
        minimum of dominant frequency (cepstrum),
        maximum of dominant frequency (cepstrum),
        modulation index,
        fundamental frequency 1,
        fundamental frequency 2,
        .....,
        fundamental frequency 30
        """
        props = []

        mean = np.mean(frq)
        props.append(mean)

        sd = np.std(frq)
        props.append(sd)

        median = np.median(frq)
        props.append(median)

        self.q75, self.q25 = np.percentile(frq, [self.q75, self.q25])
        props.append(self.q25)
        props.append(self.q75)

        iqr = self.q75 - self.q25
        props.append(iqr)

        skw = skew(frq)
        props.append(skw)

        kurt = kurtosis(frq)
        props.append(kurt)

        entr = entropy(frq)
        props.append(entr)

        flatness = gmean(Y) / np.mean(Y)
        props.append(flatness)

        peakf = esp_frecuencia_pairs[0][1]
        props.append(peakf)

        mindom = min(frq)
        props.append(mindom)

        maxdom = max(frq)
        props.append(maxdom)

        dfrange = maxdom - mindom
        props.append(dfrange)

        modindx = self.calculate_modulation_index(frq, mindom, maxdom, dfrange)
        props.append(modindx)

        # ///////// CON EL CEPSTRUM ////////////
        ceps_mean = np.mean(ceps)
        props.append(ceps_mean)

        ceps_max = max(ceps)
        props.append(ceps_max)

        ceps_min = min(ceps)
        props.append(ceps_min)

        fundamental = self.get_n_fundamental_frequencies(
            self.c, esp_frecuencia_pairs)
        print("p:", props)
        print("f", fundamental)
        return np.array(props + fundamental)

    def preprocess(self, fs, signal):
        """
        Preprocess the de audio signal.

        :param fs: Integer that contains the sampling frequency
        :param signal: Array that contains the samples in the time domain
        of the audio signal

        :return : Array with all the main features of the audio signal
        """
        # NUMERO TOTAL DE MUESTRAS
        mt = signal.size
        print("NUMERO DE MUESTRAS EN EL TIEMPO : ", mt)
        Y, frq, ceps = self.calculate_spectrum_cepstrum(signal, fs)
        print("NUMERO DE MUESTRAS EN EL LA FRECUENCIA : ", frq.size)
        # Hacemos lista de (decibeles(Y), frecuencia(x)) tuples
        esp_frecuencia_pairs = [(Y[i], frq[i]) for i in range(len(Y))]
        # APLICAMOS FILTRO DE FRECUENCIAS
        esp_frecuencia_pairs = [(Y[i], frq[i]) for i in range(
            len(Y)) if frq[i] > self.min_frq and frq[i] < self.max_frq]
        print(
            "NUMERO DE MUESTRAS EN EL LA FRECUENCIA DESPUES DE FLITRO : ",
            len(esp_frecuencia_pairs))
        # FRECUENCIAS FILTRADAS
        esp_aux = np.array(esp_frecuencia_pairs)
        frq = esp_aux[:, 1]
        # ESPECTRO DE POTENCIA
        Y = esp_aux[:, 0]
        # ORDENAMOS
        esp_frecuencia_pairs.sort()
        esp_frecuencia_pairs.reverse()
        # OBTENEMOS TODAS LAS CARACTERISTICAS RELEVANTES
        result = self.calculate_all_spec_props(
            frq, Y, ceps, esp_frecuencia_pairs)
        return result

    # ///////////////////////////////////////////////////////
    # //////////// PROCESAMIENTO Y CLASIFICACION ////////////
    # ///////////////////////////////////////////////////////

    def process(self, fs, signal):
        """
        Preprocess and clasify the de audio signal.

        :param fs: Integer that contains the sampling frequency
        :param signal: Array that contains the samples in the time domain
        of the audio signal

        :return :Dictionary with the result of the classification
        """
        response = {
            "data": {
                "gender": {
                    "id": "",
                    "name": ""
                },
                "detectionDate": ""
            }
        }
        signal = self.preprocess(fs, signal)
        signal = np.array(signal).reshape((1, -1))
        pred_xgb = self.model.predict(signal)
        result = pred_xgb[0].strip()
        result = 'FEMALE' if result == 'F' else 'MALE'

        response["data"]["gender"]["id"] = result
        response["data"]["gender"]["name"] = ("The gender of this person is : "
                                              + result)
        response["data"]["detectionDate"] = date.today().isoformat()
        return response
