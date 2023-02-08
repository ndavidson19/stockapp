import numpy as np


def EWMA(data, span=10, alpha=None):
    """
    Exponentially weighted moving average
    """
    if alpha is None:
        alpha = 2 / (span + 1)
    return np.array([data[:i + 1].dot(np.power(alpha, np.arange(i + 1)[::-1])) for i in range(len(data))])

def SMA(data, span=10):
    """
    Simple moving average
    """
    return np.convolve(data, np.ones(span), 'valid') / span

def WMA(data, span=10):
    """
    Weighted moving average
    """
    weights = np.arange(1, span + 1)
    return np.convolve(data, weights, 'valid') / weights.sum()

def HMA(data, span=10):
    """
    Hull moving average
    """
    wma = WMA(data, span)
    wma2 = WMA(wma, span)
    wma3 = WMA(wma2, span)
    return (2 * wma) - wma2 + (2 * wma3)

def DEMA(data, span=10):
    """
    Double exponential moving average
    """
    ema = EWMA(data, span)
    ema2 = EWMA(ema, span)
    return 2 * ema - ema2

def TEMA(data, span=10):
    """
    Triple exponential moving average
    """
    ema = EWMA(data, span)
    ema2 = EWMA(ema, span)
    ema3 = EWMA(ema2, span)
    return 3 * ema - 3 * ema2 + ema3

def TRIMA(data, span=10):
    """
    Triangular moving average
    """
    return SMA(SMA(SMA(data, span), span), span)

def KAMA(data, span=10, pow1=2, pow2=30):
    """
    Kaufman adaptive moving average
    """
    change = np.abs(data - np.roll(data, 1))
    volatility = np.zeros(len(data))
    volatility[span:] = change[span:].cumsum() / span
    volatility[:span] = volatility[span]
    ER = change / volatility
    sc = np.power(ER * (2 / (pow1 + 1) - 2 / (pow2 + 1) + 1), 2)
    return EWMA(data, span, sc)

def ZLEMA(data, span=10):
    """
    Zero lag exponential moving average
    """
    return EWMA(data, span)[span - 1:]

def VWMA(data, volume, span=10):
    """
    Volume weighted moving average
    """
    return np.convolve(data * volume, np.ones(span), 'valid') / np.convolve(volume, np.ones(span), 'valid')

def VAMA(data, volume, span=10):
    """
    Volume adjusted moving average
    """
    return np.convolve(data * volume, np.ones(span), 'valid') / np.convolve(volume, np.ones(span), 'valid')

def T3(data, span=10, vfactor=0.7):
    """
    Triple exponential moving average (T3)
    """
    ema = EWMA(data, span)
    ema2 = EWMA(ema, span)
    ema3 = EWMA(ema2, span)
    return ema + vfactor * (ema - ema2) + vfactor ** 2 * (ema - 2 * ema2 + ema3)



        