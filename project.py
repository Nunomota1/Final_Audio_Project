import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import wavfile
from scipy.fft import fft
import librosa
import librosa.display

# Definição do ficheiro áudio a analisar
inpfile = 'OBOESOLO_AE.wav'

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
win = np.sin(np.pi / N * np.arange(N))

# 1_b)_i)
# Reading/Recording
# Lê o áudio, obtém frequência de amostragem (FS) e guarda samples do áudio (detain)
FS, datain = wavfile.read(inpfile)
# Transforma os dados em vírgula flutuante
datain = datain.astype(np.float32)
# Converte para a escala típica 16-bit
datain = datain * 32768.0

# Quando se houve música podemos ouvir música do lado esquerdo e direito, ou seja, há dois canais
# O Python guarda os ficheiros como: [[x,y], ... ]
# Verifica se o áudio tem mais de um canal, se sim, faz a média
# O conteúdo espectral é mais importante que o espacial, ou seja, mais que saber de onde vem
# a informação, temos de saber a melodia, frequência que existem nos canais -> MFCC foca-se no conteúdo espectral e não
# a localização)
# Além disso, garantimos que só há um tipo de dados, que não haja informação redundante (visto
# que as informações dos canais são semelhantes
if len(datain.shape) > 1:
    datain = np.mean(datain, axis=1)

# Janela gráfico
plt.figure(1, figsize=(12, 5))
plt.subplot(1, 2, 1)

# PARÂMETROS: frequência de amostragem, janela, cada frame tem 1024 samples, overlap de 512
# (50% de overlap), FFT com tamanho de 1024 e queremos a magnitude do espetro
# SSX é uma matriz, onde as linhas representam frequências, as colunas são intantes 
# temporais e os valores são a energia/intensidade
frequencies, times, Sxx = signal.spectrogram(datain, fs=FS, window='hann', nperseg=N, noverlap=N2, nfft=N, mode='magnitude' )

# Converte para dB e usamos o logaritmo porque o ouvido humano é logaritmico e percebe
# diferenças relativas e não absolutas
Sxx_db = 20 * np.log10(Sxx + 1e-10)

# Desenha o espetrograma: eixo x (tempo), eixo y (frequência) e cor (energia)
plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud')
plt.ylim([0, FS / 2])
plt.clim([-40, 50])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('LIN Spectrogram')
plt.subplot(1, 2, 2)

# 1_b)_v)
# Escala MEL imita a perceção humana
# n_mels = 128 -> diz o número de filhos MEL 
# power = 2.0 -> calcula [FFT]^2 e calcula a potência espectral
# melspectrogram faz automaticamente framing, janela, FFT, power spectrum e MEL filterbanck
mel_spec = librosa.feature.melspectrogram(y=datain, sr=FS, n_fft=N, hop_length=N2, win_length=N, window='hann', n_mels=128, power=2.0)

# 1_b)_vi)
# Converte potência em decibeis
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Mostra o especrogram MEL
# Tem mais detalhe em baixas frequênciae e menos resolução nas altas frequências
# Como o ouvido humano
librosa.display.specshow( mel_spec_db, sr=FS, hop_length=N2, x_axis='time', y_axis='mel')

# Mostra o espetrograma MEL
plt.clim([-40, 50])
plt.title('MEL Spectrogram')
plt.tight_layout()
plt.show()

# Número de amostras de áudio
nread = len(datain)

# Dá print à frequência de amostragem
print("Sampling frequency:")
print(FS)

# Inicializa o frame e define qual está a analisar (o primeiro)
frame = 1
# Região de frequência: vai apenas até N/2 porque a FFT de sinais é simétrica
# (ínicio positivo e fim negativo)
regiaofreq = np.arange(0, N2 + 1)
# Percorre o áudio frame a frame até houver samples suficientes
while ((frame + 1) * N2 < nread):
    # 1_b)_iii)
    # frame 1 -> começa em 0
    # O segundo começa em 512
    varre_start = (frame - 1) * N2
    # O frame 1 acaba em 1024 e o segundo em 1536
    # Logo cada frame tem 1024 samples
    varre_end = (frame + 1) * N2
    # Extrair frame do áudio -> seleciona um segmento do áudio
    tmpdata = datain[varre_start:varre_end]
    # No final do áudio pode não haver samples suficientes e verifica se o último 
    # tem 1024 samples, se sim, o loop acaba para evitar erros no processamento, isto porque
    # todos os blocos têm tamanho fixo
    if len(tmpdata) < N:
        break
    # Primeiro grafico
    plt.figure(2, figsize=(12, 5))
    # Divide em esquerda e direita
    plt.subplot(1, 2, 1)
    # Mostra waveforms no domínio temporal
    # Eixo X: número de samples
    # Eixo do y: ampltiude
    # Isto representa como o sinal muda ao longo do tempo 
    plt.plot(np.arange(N), tmpdata[:N])
    plt.xlabel('Samples')
    plt.ylabel('Normalized amplitude')
    plt.title('Time segment of the signal')
    # Multiplica o frame pela janela sinusoidal -> suaviza as extremidades

    # 1_b)_iii)
    tmpdata_win = tmpdata[:N] * win
    # Transforma o tempo -> em frequência

    # 1_b)_iv)
    # Transforma o tempo -> em frequência
    fdata = fft(tmpdata_win)

    # 1_b)_iv)
    # FFT devolve números complexos e assim representamos a intensidade das frequências
    # np.finfo(float).eps -> é um número muito pequeno para evitar log(0)
    magnitude = np.abs(fdata) + np.finfo(float).eps

    # Extrai a fase da FFT
    # FFT tem magnitude (intensidade) e fase (deslocamento)
    # MFCC usa magnitude
    fase = np.angle(fdata)

    # Desenha o espectro
    plt.subplot(1, 2, 2)
    # Converte FFT em Hz
    freq_axis = FS / N * regiaofreq

    # Mostra o espetro da magnitude
    # Eixo X: frequência
    # Eixo Y: energia em dB
    plt.plot(
        freq_axis,
        20 * np.log10(magnitude[regiaofreq])
    )
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral Density (dB)')
    plt.title('Magnitude spectrum of the segment')
    plt.tight_layout()
    plt.show()
    input("Press ENTER to continue...")
    frame += 1
print('End of processing.')