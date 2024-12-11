#!/bin/bash                                                                                                                
echo 'Date: ' $(date)                                                                                                      
echo 'Host: ',$(hostname)                                                                                                  
echo 'System: ' $(uname -spo)                                                                                              
echo 'Home: ', $HOME                                                                                                       
echo 'Worddir: ',$PWD                                                                                                      
echo "Program: $0"                                                                                                         
echo "Args: $*"                                                                                                            

#source /cvmfs/sft.cern.ch/lcg/views/LCG_101_ATLAS_3/x86_64-centos7-gcc8-opt/setup.sh                                       
#source  /cvmfs/sft.cern.ch/lcg/views/LCG_102b//x86_64-centos8-gcc11-opt/setup.sh                                           
source /cvmfs/sft.cern.ch/lcg/views/LCG_101_ATLAS_3/x86_64-centos7-gcc8-opt/setup.sh

export PYTHONPATH=$PYTHONPATH:$USER/.numba                                                                            

echo 'PYTHONPATH: ' $PYTHONPATH                                                                                            
echo 'Path: ' $PATH  
python3  /afs/cern.ch/user/m/mbressan/private/ising/main.py $1 $2
