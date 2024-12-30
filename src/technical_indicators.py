# Technical indicators calculation functions
def calculate_rsi(prices, periods=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def calculate_atr(data, period=14):
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_momentum(data):
    """
    Calculate momentum as the cumulative sum of the first derivative 
    of the 5-period moving average over the past 5 periods
    """
    # Calculate first derivative of 5-period MA
    derivative = data['SMA_5'].diff()
    
    # Calculate rolling sum of derivatives over past 5 periods
    cumulative_momentum = derivative.rolling(window=5).sum()
    
    return cumulative_momentum