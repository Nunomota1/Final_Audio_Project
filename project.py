import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import wavfile
from scipy.fft import fft
from scipy.fftpack import dct
import librosa

# 1_b)_ii)
# Permite a divisão em frames de 40 ms
# N = comprimento da transformada DCT
# 1024/22050 = 0.46 ms, ou seja, perto de 40 ms de frames
N = 1024

# 1_b)_iii)
# Overlap de 50% -> 512 samples do anterior
# N/2 = sobreposição entre tramas (overlap de 50%)
N2 = N // 2

# 1_b)_iii)
# Windowing
# win = janela de análise
# A função seno começa em 0, sobre suavemente e desce suavemente para 0, logo, o frame
# deixa de acabar abruptamente
# Suaviza as bordas
win = np.sin(np.pi / N * np.arange(N))

# Dicionário onde vamos guardar (nome da música:fingerprint)
data = {}

# Função responsável por obter a fingerprint da música de 30 segundos 
def fingerprint(inpfile):

    # 1_b)_i)
    # Reading/Recording
    # Lê o áudio, obtém frequência de amostragem (FS) e guarda samples do áudio (detain)
    FS, data_sinal = wavfile.read(inpfile)
    # Transforma os dados em float
    data_sinal = data_sinal.astype(np.float32)
    # Coloca o áudio entre -1 e 1 e também evita problemas com o logaritmo (evita o log(0))
    data_sinal = data_sinal / (np.max(np.abs(data_sinal)) + 1e-10)

    # Quando se houve música podemos ouvir música do lado esquerdo e direito, ou seja, há dois canais
    # O Python guarda os ficheiros como: [[x,y], ... ]
    # Verifica se o áudio tem mais de um canal, se sim, faz a média
    # O conteúdo espectral é mais importante que o espacial, ou seja, mais que saber de onde vem
    # a informação, temos de saber a melodia, frequência que existem nos canais -> MFCC foca-se no conteúdo espectral e não
    # a localização)
    # Além disso, garantimos que só há um tipo de dados, que não haja informação redundante (visto
    # que as informações dos canais são semelhantes
    if len(data_sinal.shape) > 1:
        data_sinal = np.mean(data_sinal, axis=1)

    # Simulação do ouvido humano 
    # Com 128 bandas de frequência
    mel_filterbank = librosa.filters.mel(sr = FS, n_fft = N, n_mels = 128)

    # Guarda MFCC de cada frame
    mfcc_list = []

    # Divide o áudio em frames
    # len(data_sinal): garante que há sempre frames completos a ser analisados
    for i in range(0, len(data_sinal) -  N, N2):
        # Extrai um frame
        frame = data_sinal[i:i+N]
        # Grante que o tamanho do frame é 1024, ou seja, não aceita frames incompletos
        if len(frame) < N:
            break
        # Aplica Windowing
        win_aux = frame * win
        # FFT
        # Aplica FFT (tempo -> frequência)
        # Apenas acede aos valores da magnitude - exclusivamente a quantidade de cada frequência no som
        # Metade do espetro, porque a outra metade é redundante
        magnitude_spectrum = np.abs(fft(win_aux))[:N//2 + 1]
        # Tranforma o espetro de magnitude em energia
        power_spectrum = (magnitude_spectrum ** 2)
        # MEL FILTERBANK
        # Transforma o espetro numa representação que o ouvido humano não consegue ouvir
        # Analisa frequência separadas
        # Agrupa frequências semelhantes
        mel_energies = np.dot(mel_filterbank, power_spectrum)
        # LOG
        # Compressão de energia
        # O ouvido humano funciona com o logaritmo
        # 1e-10 -> evita o erro do log(0)
        log_mel = np.log(mel_energies + 1e-10)
        # MFCC
        mfcc = dct(log_mel, type=2, norm='ortho')[:13]
        # Guarda MFCC do frame em questão
        mfcc_list.append(mfcc)

    #  Transforma em matriz
    mfcc_list = np.array(mfcc_list)

    # Comportamento preventivo caso não haja matrizes
    if len(mfcc_list) == 0:
        return np.zeros(13), mfcc_list, data_sinal, FS
    # FINGERPRINT
    # Média de todos os frames
    fingerprint = np.mean(mfcc_list, axis=0)
    # Return do fingerprint, MFCC por frame, áudio e a frequência de amostragem
    return fingerprint, mfcc_list, data_sinal, FS

filepath = "songs_database/"

# Percorre a base dos clipes de 30 segundos 
for audio in os.listdir(filepath):
    # Condição de como deve terminar o ficheiro
    if audio.endswith(".wav"):
        full_path = os.path.join(filepath, audio)
        print(f"Audio: {audio}")
        # Cálculo do fingerprint do áudio
        value_fingerprint, mfcc, data_sinal, FS = fingerprint(full_path)
        # Guarda no dicionário o fingerprint da música
        data[audio] = value_fingerprint
        # Visualização do fingerprint do áudio
        plt.figure(figsize=(10, 4))
        # Troca as linhas -> ou seja, coeficientes em função do tempo
        # Eixo X: Frames
        # Eixo Y: MFCC
        plt.imshow(mfcc.T, aspect='auto', origin='lower')
        plt.title(f"MFCC (manual pipeline): {audio}")
        plt.xlabel("Frames")
        plt.ylabel("MFCC Coefficients")
        plt.colorbar()
        plt.show(block=False)
        plt.pause(2)
        plt.close()