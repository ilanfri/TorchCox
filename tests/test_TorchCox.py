from torchcox import TorchCox
import pytest
import pandas as pd
import numpy as np

valdf = pd.DataFrame({'id':['Bob','Sally','James','Ann'], 'time':[1,3,6,10], 'status':[1,1,0,1], 'smoke':[1,0,0,1]})

def test_TorchCox():

    tname = 'time'
    Xnames = ['smoke']
    dname = 'status'
    
    coxmod = TorchCox.TorchCox()
    coxmod.fit(valdf, Xnames=Xnames, tname=tname, dname=dname)
    
    beta = coxmod.beta.detach().numpy()
    
    assert np.log(2)/2 == pytest.approx(beta, abs=1e-5)
    
