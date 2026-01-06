
def to_numeric(X):
  X=X.copy()
  for col in X.columns:
    X[col]=pd.to_numeric(X[col],errors='coerce')
  return X

def clean_internet(X):
  X=X.replace({'No internet service':'No'})
  return X
