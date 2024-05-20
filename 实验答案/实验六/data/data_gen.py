#!/usr/bin/env python
###################################################################
# File Name: data_gen.py
# Author: Li Zhen
# Email: lizhen2014@ict.ac.cn
# Created Time: 2020-02-13 09:24:58 CTS
# Description: 
###################################################################
from os import sys
import random as rd
import math as mh

def num2str(num, mode, width):
  str_ori = ""
  # bin
  if(mode == 0):
    str_ori = bin(num)[2:] # get rid of 0b
  else:
    str_ori = hex(num)[2:] # get rid of 0x

  str = ""
  if(len(str_ori) < width):
    str = (width - len(str_ori)) * "0" + str_ori
  elif (len(str_ori) > width):
    str = str_ori[-width:] # truncate the result to last width bits
  else:
    str = str_ori

  return str

def verctor_gen(pfr, nfr, sfr, iter):
  iter0   = rd.randint(0,15)
  iter1   = rd.randint(0,15)
  
  line_iter = 32
  base      = 2**16
  width     = 4

  partsum = 0
  for i in range(iter):
    neu_str = ""
    syn_str = ""
  
    for k in range(line_iter):
      # random generate complement form of number
      neu = rd.randint(0,2**16) % base
      syn = rd.randint(0,2**16) % base
      neu_str = neu_str + num2str(neu, 1, width) 
      syn_str = syn_str + num2str(syn, 1, width) 
  
      # convert the complement form to the true value for signed multiply
      if(neu >= (base/2)): neu = neu - base
      if(syn >= (base/2)): syn = syn - base

      # multiply, get the true value of mul
      mul = neu * syn
      if(mul < 0): mul += 2**32 # convert mul to complement form for add to partsum

      partsum += mul
      partsum = partsum & 0xFFFFFFFF # partsum is 32-bit, get rid of overflow

    nfr.write(neu_str+"\n")
    sfr.write(syn_str+"\n")
  
  pfr.write(num2str(partsum, 0, 32)+"\n")


#if __name__ == "__main_":
nfr = open("neuron", "w+")
sfr = open("weight", "w+")
pfr = open("result", "w+")

# 20 + 30 + 40 + 50 = 140
verctor_gen(pfr, nfr, sfr, 20)

verctor_gen(pfr, nfr, sfr, 30)

verctor_gen(pfr, nfr, sfr, 40)

verctor_gen(pfr, nfr, sfr, 50)
