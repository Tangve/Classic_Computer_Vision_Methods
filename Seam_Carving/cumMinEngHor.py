'''
  File name: cumMinEngHor.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the horizontal seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - OUTPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
'''

from cumMinEngVer import cumMinEngVer
import numpy as np

def cumMinEngHor(e):
    E=np.copy(e)
    E=E.T
    MyT,TbyT=cumMinEngVer(E)
    My=MyT.T
    Tby=TbyT.T
    return My, Tby