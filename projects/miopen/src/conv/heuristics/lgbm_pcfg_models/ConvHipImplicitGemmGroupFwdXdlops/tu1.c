
#include "header.h"

void predict_unit1(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.161602735519410068) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.004101839504132475;
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.019479732232153008;
          } else {
            result[0] += -0.007754364409384378;
          }
        }
      } else {
        result[0] += -0.009358336642543133;
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.0037314569622010265;
          } else {
            result[0] += -0.07148454536282794;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.016539223408214718;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.02802361046016206;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.66412305831909357) ) ) {
                  result[0] += 0.008509611478359783;
                } else {
                  result[0] += -0.019547796441433493;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.026518608583012273;
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.005696016879709151;
                } else {
                  result[0] += 0.08683849352403361;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += 0.06364056219980187;
        } else {
          result[0] += -0.04512202578095667;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.473471879959107333) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.027856961447736348;
          } else {
            result[0] += -0.08295850296993197;
          }
        } else {
          result[0] += 0.00751886143321406;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += 0.026623344581113526;
            } else {
              result[0] += -0.0008004993693151677;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.869292974472046787) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.014687943757875632;
                } else {
                  result[0] += -0.010634296538129702;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.01327938062954473;
                } else {
                  result[0] += -0.036367564062280415;
                }
              }
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.004855184814494044;
              } else {
                result[0] += 0.008991105404501166;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.011115805698356837;
          } else {
            result[0] += -0.056914286865965;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40695333480835139) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.012442196842031074;
              } else {
                result[0] += -0.04732401824864905;
              }
            } else {
              result[0] += -0.002382546661060521;
            }
          } else {
            result[0] += 0.0042817231338959025;
          }
        } else {
          result[0] += -0.035237999937568605;
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.01767026158008353;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.05656005873547644;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.04044319533931858;
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.02371058542071096;
                  } else {
                    result[0] += -0.01781837693518901;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.037217342642309574;
              } else {
                result[0] += -0.010988328669207308;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.025305092114451352;
                } else {
                  result[0] += -0.026172806980288982;
                }
              } else {
                result[0] += 0.09130836730963679;
              }
            } else {
              if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.03249793945584881;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.029978893439561463;
                  } else {
                    result[0] += 0.045646517311896256;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.018716470299798758;
                } else {
                  result[0] += -0.006728628127338936;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.66412305831909357) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.015246749468606366;
                    } else {
                      result[0] += -0.00868467732926902;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += -0.0010819752556242995;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                        result[0] += -0.07080142898424308;
                      } else {
                        result[0] += -0.02285008175380003;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += -0.008228948748877602;
                    } else {
                      result[0] += 0.02471582865028419;
                    }
                  } else {
                    result[0] += -0.0015898939356441032;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.020454799105356344;
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0009949858778577137;
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.12554320253914278;
                      } else {
                        result[0] += 0.017247455020763372;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.0415304173566268;
                    } else {
                      result[0] += 0.00700102384554542;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.013401015917469145;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.161602735519410068) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.0038370709878546997;
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.018534239565597913;
          } else {
            result[0] += -0.006739194799535342;
          }
        }
      } else {
        result[0] += -0.008780282884125806;
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.0042284585442341335;
          } else {
            result[0] += 0.0076371201870078015;
          }
        } else {
          result[0] += -0.01570630344195361;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += 0.056826643549821455;
        } else {
          result[0] += -0.04205855978298431;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.03338779913904845;
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.0032056040914916657;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            result[0] += -0.05888572112967947;
          } else {
            result[0] += -0.014443353949604262;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.772694945335388628) ) ) {
                result[0] += -0.03811744260386395;
              } else {
                result[0] += 0.014028614559336797;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.056210605448833065;
              } else {
                result[0] += 0.00997938966435167;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.0025544424184160428;
              } else {
                result[0] += -0.021257498608399605;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.03970021458650367;
              } else {
                result[0] += -0.008668604821463708;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
            result[0] += -0.019057050790904962;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += 0.023531463994410536;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.490982532501221591) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.003525148829504814;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.01344189260638889;
                      } else {
                        result[0] += 0.038042434323195005;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                        result[0] += 0.011329076873608013;
                      } else {
                        result[0] += 0.042635058965994344;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.027506700163559446;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                        result[0] += 0.030672303679875924;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                            result[0] += -0.003518613360959955;
                          } else {
                            result[0] += -0.04039777757962842;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                            result[0] += 0.010956279808764;
                          } else {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.10147598377641454;
                              } else {
                                result[0] += -0.009121816999525956;
                              }
                            } else {
                              result[0] += 0.03979285220111634;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.0634677637658636;
                    } else {
                      result[0] += 0.07697273184397584;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.0041947649297501485;
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.03202800657358405;
                      } else {
                        result[0] += 0.059749298725070635;
                      }
                    }
                  } else {
                    result[0] += -0.01272075577904343;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.008509288565524646;
                    } else {
                      result[0] += -0.029237808226929216;
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.02785608792267948;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.134879350662232333) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.149475097656251776) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            result[0] += 0.011838060977773596;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
                              result[0] += -0.059888433566396185;
                            } else {
                              result[0] += -0.003838477111792197;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                              result[0] += -0.06225640276007205;
                            } else {
                              result[0] += 0.0015247095512257094;
                            }
                          } else {
                            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += 0.028348124187885772;
                            } else {
                              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                                result[0] += -0.018385446916043418;
                              } else {
                                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.7800335884094256) ) ) {
                                  result[0] += 0.014486227535000097;
                                } else {
                                  result[0] += 0.11260014158629182;
                                }
                              }
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.0071997969383891455;
                        } else {
                          result[0] += 0.04054794496803398;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.005394386603509101;
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.05840526081314035;
                } else {
                  result[0] += -0.009641472887148943;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.791781663894654208) ) ) {
                  result[0] += -0.025729914435901287;
                } else {
                  result[0] += 0.020189468206042747;
                }
              }
            }
          } else {
            result[0] += -0.0011168160416837986;
          }
        } else {
          result[0] += 0.012843177524805537;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.014576081208680529;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
            result[0] += -0.0003377348693766304;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.014710860035132896;
                } else {
                  result[0] += -0.04017223474179747;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += 0.0646993163186395;
                    } else {
                      result[0] += -0.06292584133265006;
                    }
                  } else {
                    result[0] += -0.05416770944552596;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.0368269294546136;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                      result[0] += -0.008178077493351787;
                    } else {
                      result[0] += 0.01282428744012456;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.020001374507655715;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.031229972598857042;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += 0.011932482874562679;
                    } else {
                      result[0] += -0.004486500403285646;
                    }
                  } else {
                    result[0] += -0.03539183624626236;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.0012541750791809521;
                      } else {
                        result[0] += -0.01568040130645773;
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.566809177398682529) ) ) {
                        result[0] += 0.006775431629835834;
                      } else {
                        result[0] += -0.03140516515741134;
                      }
                    }
                  } else {
                    result[0] += 0.024627120657272832;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.169590950012207919) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
                          result[0] += -0.00531270373273934;
                        } else {
                          result[0] += 0.014113330781027129;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                          result[0] += -0.04597159647137767;
                        } else {
                          result[0] += -0.0033953330616367274;
                        }
                      }
                    } else {
                      result[0] += 0.033232667478302044;
                    }
                  } else {
                    result[0] += -0.030707792250023513;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.561026811599732333) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40695333480835139) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0018625921886933471;
                      } else {
                        result[0] += 0.009933518317625024;
                      }
                    } else {
                      result[0] += 0.017760317131672435;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.054856408234274005;
                      } else {
                        result[0] += -0.02360265288013127;
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.03435927197798327;
                      } else {
                        result[0] += -0.0035795552724579393;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                    result[0] += 0.002291366289206843;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.05693568343717209;
                      } else {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.04042782259980854;
                        } else {
                          result[0] += 0.006977427036908923;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.874179124832154208) ) ) {
                            result[0] += -0.013351976153951282;
                          } else {
                            result[0] += -0.0511939316199028;
                          }
                        } else {
                          result[0] += 0.016086921032432883;
                        }
                      } else {
                        result[0] += 0.012093725797889013;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.558514595031739169) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.07533909338667572;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.01483916007232651;
                  } else {
                    if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
                            result[0] += 0.014588632950829867;
                          } else {
                            result[0] += -0.029343039634456988;
                          }
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += 0.021174753223478636;
                          } else {
                            result[0] += 0.06249149337848178;
                          }
                        }
                      } else {
                        result[0] += -0.0008690001260720334;
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.020127415657043901) ) ) {
                        result[0] += -0.003597376607209995;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += 0.016559037851274626;
                        } else {
                          result[0] += -0.07443107549680725;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.052533896634149216;
                  } else {
                    result[0] += -0.012924849603428795;
                  }
                } else {
                  result[0] += 0.03841832833509449;
                }
              }
            }
          } else {
            result[0] += 0.03221848895623286;
          }
        }
      }
    } else {
      result[0] += -0.04257608052274475;
    }
  } else {
    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.7619357109069842) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
          result[0] += -0.018783049462669996;
        } else {
          result[0] += 0.0632229088416025;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
          result[0] += -0.03340301049418758;
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.034886208971451275;
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.02104736656313261;
            } else {
              result[0] += 0.14908966110520716;
            }
          }
        }
      }
    } else {
      result[0] += 0.08561670427537214;
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
      result[0] += 0.014783879245036938;
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.02108009147339174;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.932935476303101474) ) ) {
                    result[0] += -0.08801235964833809;
                  } else {
                    result[0] += 0.09130814950605846;
                  }
                }
              } else {
                result[0] += 0.008460480270692579;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.645421981811524326) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    result[0] += -0.005688799471386693;
                  } else {
                    result[0] += 0.019126378015436457;
                  }
                } else {
                  result[0] += -0.021530398456607664;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += -0.029298094701761812;
                } else {
                  result[0] += -0.0018904627841888942;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.0009398644404695671;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += 0.03677841691373144;
              } else {
                result[0] += -0.12441137135074623;
              }
            }
          }
        } else {
          result[0] += 0.006092113576110618;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.847910165786744052) ) ) {
                      result[0] += 0.006453843226574313;
                    } else {
                      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.008541822094764683;
                      } else {
                        result[0] += 0.02704356097786232;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.015967625648540002;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.0883522033691424) ) ) {
                          result[0] += -0.007723570042453078;
                        } else {
                          result[0] += 0.02847509791064446;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                        result[0] += -0.08053288098249958;
                      } else {
                        result[0] += -0.0157339690910693;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                            result[0] += -0.008778048647835343;
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.797939777374268466) ) ) {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                                result[0] += -0.053857186588379624;
                              } else {
                                result[0] += -0.026601686867026277;
                              }
                            } else {
                              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.04909609112094678;
                              } else {
                                result[0] += 0.009848782910428278;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.02308977003708807;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                          result[0] += -0.007125602735115491;
                        } else {
                          result[0] += 0.036754555407264876;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.169590950012207919) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.087577104568482333) ) ) {
                            result[0] += 0.03363181895069982;
                          } else {
                            result[0] += 0.00891781769262259;
                          }
                        } else {
                          result[0] += 0.003496920480320398;
                        }
                      } else {
                        result[0] += -0.007201886880522573;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.003838300704956943) ) ) {
                        result[0] += -0.0021881911626633263;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += 0.000761057317503204;
                                } else {
                                  result[0] += -0.04612877801708302;
                                }
                              } else {
                                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                    result[0] += 0.00857241752306933;
                                  } else {
                                    result[0] += 0.04419354407743473;
                                  }
                                } else {
                                  result[0] += -0.015171020913073678;
                                }
                              }
                            } else {
                              result[0] += 0.033851732559445735;
                            }
                          } else {
                            result[0] += -0.039392018479106;
                          }
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.018509280514959044;
                          } else {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                              result[0] += 0.025003957599403128;
                            } else {
                              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += -0.00038121615970294393;
                              } else {
                                result[0] += -0.11701223050131876;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                        result[0] += 0.007034126757230661;
                      } else {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.802100181579590732) ) ) {
                              result[0] += -0.012822777865418258;
                            } else {
                              result[0] += -0.05917514215414891;
                            }
                          } else {
                            result[0] += -0.10983251169198081;
                          }
                        } else {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.031169482367916036;
                          } else {
                            result[0] += 0.004818250664299844;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0068775970787635955;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.702574253082276279) ) ) {
                result[0] += -0.010806569333187402;
              } else {
                result[0] += 0.0032408661135570303;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.11219056516144875;
            } else {
              result[0] += 0.018639844777226668;
            }
          }
        } else {
          result[0] += -0.0008043889460656732;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.07293939590454279) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        result[0] += 0.0256946393479087;
      } else {
        result[0] += -0.032402332901229604;
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.007665888248490122;
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
          result[0] += -0.038480263519576646;
        } else {
          result[0] += 0.1181502711574277;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.319199085235596591) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
              result[0] += -0.02888173537909891;
            } else {
              result[0] += 0.00564280005960745;
            }
          } else {
            result[0] += 0.039364566502156056;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.551017761230469638) ) ) {
            result[0] += 0.005889179573369027;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.006108247887637801;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.031211975816072227;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.613121509552002841) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.04792280990144735;
                  } else {
                    result[0] += 0.017690948336580994;
                  }
                } else {
                  result[0] += 0.023906643249150927;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.0013284848536986495;
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.0802530335733831;
          } else {
            result[0] += 0.0010901628586340018;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.463808774948121005) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.014116031263878441;
              } else {
                result[0] += -0.044641772352789516;
              }
            } else {
              result[0] += -0.0017942946126665825;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              result[0] += -0.018891978705116698;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
                result[0] += -0.02657485346185951;
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.06928695144133067;
                } else {
                  result[0] += 0.060316609794524;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.09939996109205251;
              } else {
                result[0] += 0.0016616320813830653;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
                  result[0] += -0.019668324465338668;
                } else {
                  result[0] += -0.046930332727939984;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                  result[0] += -0.020768565725721305;
                } else {
                  result[0] += 0.01674270154523397;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.005890483496623582;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.03005934970447907;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.86392068862915217) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                          result[0] += 0.0441711010055428;
                        } else {
                          result[0] += -0.07395474807515966;
                        }
                      } else {
                        result[0] += 0.005630617248651774;
                      }
                    } else {
                      result[0] += 0.07148020774646378;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.02661483346608185;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += -0.03646728877493592;
                      } else {
                        result[0] += 0.015565970421027775;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.08457948277376587;
                    } else {
                      result[0] += -0.01486653178455631;
                    }
                  } else {
                    result[0] += 0.007203500178290448;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.03513007684328592;
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.015036822760737034;
                    } else {
                      result[0] += 0.010141066882282376;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.018345004194822358;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.021354682525004373;
                        } else {
                          result[0] += 0.054000997548280465;
                        }
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                              result[0] += 0.039901375636070484;
                            } else {
                              result[0] += -0.03186857697504195;
                            }
                          } else {
                            result[0] += 0.07307070152577112;
                          }
                        } else {
                          result[0] += 0.07581754206276925;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.17202329635620295) ) ) {
                          result[0] += -0.060886200348615416;
                        } else {
                          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.14071154808594064;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.21245336532592951) ) ) {
                              result[0] += -0.0041106831909960735;
                            } else {
                              result[0] += 0.10791928154375124;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                          result[0] += 0.05678201692929509;
                        } else {
                          result[0] += 0.014291425899988768;
                        }
                      }
                    } else {
                      result[0] += -0.026337761633873354;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.04549079920739717;
                  } else {
                    result[0] += 0.02906035434774545;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.0883984565734881) ) ) {
            result[0] += -0.08653775967279459;
          } else {
            result[0] += -0.007180659794946602;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        result[0] += 0.020267660083284414;
      } else {
        result[0] += -0.030914480358485125;
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += 0.0016652781379605998;
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
          result[0] += -0.013310655481116589;
        } else {
          result[0] += 0.11282665125996445;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[39].missing != -1) && (data[39].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.005897518032010083;
    } else {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0006714935359530368;
              } else {
                result[0] += 0.020976434507540054;
              }
            } else {
              if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.004363768077583063;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0051523647000248024;
                } else {
                  result[0] += -0.08830867448359504;
                }
              }
            }
          } else {
            result[0] += -0.014091488784993753;
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.597218394279480425) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.007962762109659903;
            } else {
              result[0] += -0.02621338567048972;
            }
          } else {
            result[0] += -0.005431555575141273;
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.01490873735693531;
              } else {
                result[0] += 0.008428254419813277;
              }
            } else {
              result[0] += -0.022294673842999067;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.018601373934319963;
                    } else {
                      result[0] += 0.0009901226106501688;
                    }
                  } else {
                    result[0] += 0.02076722845588136;
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.017460136617776106;
                      } else {
                        result[0] += -0.017181191695830326;
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
                        result[0] += -0.0021683660946800547;
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.03386375342687409;
                        } else {
                          result[0] += 0.014897738610329229;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += -0.015000384714573854;
                        } else {
                          result[0] += -0.12721690770564859;
                        }
                      } else {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += -0.11709206629184249;
                          } else {
                            result[0] += 0.006566982083039014;
                          }
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.06855079881624072;
                          } else {
                            result[0] += -0.025135810246735574;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += 0.04271844615572331;
                          } else {
                            result[0] += -0.03670311500365658;
                          }
                        } else {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += -0.1216658083880799;
                          } else {
                            result[0] += -0.01394908806882004;
                          }
                        }
                      } else {
                        result[0] += -0.010870598159548833;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.049876739949576636;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
                    result[0] += -0.0012279804218476586;
                  } else {
                    result[0] += -0.04066490408228275;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35526132583618342) ) ) {
                      result[0] += -0.014127659135041515;
                    } else {
                      result[0] += 0.01333726381916884;
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.020036413675654622;
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.01000699672628453;
                      } else {
                        result[0] += -0.030698716979684494;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.0037230979222068515;
                            } else {
                              result[0] += -0.11385248157166586;
                            }
                          } else {
                            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                              result[0] += -0.10637577202381054;
                            } else {
                              result[0] += 0.010325050291808444;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.05489025418913538;
                          } else {
                            result[0] += 0.025659680396815876;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += 0.01425911618477089;
                        } else {
                          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.018632940294418195;
                          } else {
                            result[0] += 0.008431859262759363;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += 0.047481091960864676;
                        } else {
                          result[0] += 0.01369234977051807;
                        }
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += 0.1073075610220624;
                          } else {
                            result[0] += -0.08296589373952898;
                          }
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                              result[0] += -0.01447724463159028;
                            } else {
                              result[0] += -0.1046794184590679;
                            }
                          } else {
                            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += -0.04282768688993889;
                            } else {
                              result[0] += 0.07524597704228095;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.06889040648793204;
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.09073136373693907;
                  } else {
                    result[0] += -0.028987087839558664;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
            result[0] += 0.028843775491189685;
          } else {
            result[0] += -0.011684862532970552;
          }
        }
      }
    }
  } else {
    result[0] += 0.0009174845336797999;
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.01199183455919494;
        } else {
          result[0] += 0.12701692227057462;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.00010248310292602324;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += -0.009710535600942233;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                      result[0] += -0.027259637147399148;
                    } else {
                      result[0] += 0.021922779058785855;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.726826429367066318) ) ) {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.09211965776346735;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                        result[0] += -0.1002678642514729;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.016139504694219913;
                        } else {
                          result[0] += 0.0310981354491584;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.235757827758790839) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.00617687048811647;
                        } else {
                          result[0] += 0.0577008736633479;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                          result[0] += -0.0904006000169943;
                        } else {
                          result[0] += 0.027027936311136674;
                        }
                      }
                    } else {
                      result[0] += 0.07667767253666427;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0699601680150828;
                    } else {
                      result[0] += -0.006584392681078981;
                    }
                  } else {
                    result[0] += -0.08476386997601454;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.0011673604835351792;
                  } else {
                    result[0] += -0.09845641954083675;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.00948706754207682;
                  } else {
                    result[0] += 0.044941303260563836;
                  }
                } else {
                  result[0] += 0.15773047675538115;
                }
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                        result[0] += -0.09466383654439597;
                      } else {
                        result[0] += -0.13690447393446528;
                      }
                    } else {
                      result[0] += -0.031610929052332225;
                    }
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0937684520798331;
                      } else {
                        result[0] += 0.022141915537336937;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.338562726974488193) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.257122993469240058) ) ) {
                          result[0] += -0.022806798345561537;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                            result[0] += -0.06964346390470583;
                          } else {
                            result[0] += 0.02433726765601428;
                          }
                        }
                      } else {
                        result[0] += 0.06408703826444595;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.039561872638217796;
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.436733961105347568) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.001442819524820639;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.378218650817871982) ) ) {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                              result[0] += 0.0023654394564530017;
                            } else {
                              result[0] += 0.062007775734694986;
                            }
                          } else {
                            result[0] += 0.09536431027817828;
                          }
                        }
                      } else {
                        result[0] += 0.09486562948451356;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.07380806206732725;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79285955429077326) ) ) {
                          result[0] += -0.010213553432198617;
                        } else {
                          result[0] += -0.046771588017424426;
                        }
                      } else {
                        result[0] += -0.15120075701663316;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.022131456384641965;
                  } else {
                    result[0] += 0.1437431398626123;
                  }
                } else {
                  result[0] += -0.03366882979146878;
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.1312976290411924;
                } else {
                  result[0] += -0.0638859176918021;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.581613063812256748) ) ) {
              result[0] += -0.035398498120338594;
            } else {
              result[0] += 0.06965176214688495;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.73715209960937678) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.09814652346464305;
                  } else {
                    result[0] += -0.008288195241708722;
                  }
                } else {
                  result[0] += -0.029887588076885407;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.12201496594952843;
                  } else {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.038906443573019985;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                          result[0] += 0.01777454467782764;
                        } else {
                          result[0] += -0.05124775332340601;
                        }
                      } else {
                        result[0] += 0.08977830727644942;
                      }
                    }
                  }
                } else {
                  result[0] += 0.05553790078921611;
                }
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.014380079309344275;
              } else {
                result[0] += 0.1120713082373523;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        result[0] += 0.007732809406814161;
      } else {
        result[0] += 0.07379049508369159;
      }
    }
  } else {
    result[0] += -0.036746518275878236;
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.363078355789185458) ) ) {
          result[0] += 0.048212812505717295;
        } else {
          result[0] += 0.00967797871063026;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.000115759132611318;
        } else {
          result[0] += 0.07931359469765212;
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += -0.03116321809156879;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                result[0] += -0.1433801324920085;
              } else {
                result[0] += 0.22026335785646467;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += 0.0179658792924181;
            } else {
              result[0] += -0.13122424133053576;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
            result[0] += -0.019381346566053236;
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.665046453475953037) ) ) {
              result[0] += -0.002283794833301555;
            } else {
              result[0] += 0.10764257768231096;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.04248709256340989;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.18732333183288663) ) ) {
                result[0] += 0.04432068861688573;
              } else {
                result[0] += -0.07807218862205824;
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.009773410003573988;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.007590087755019327;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.82380008697509943) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.325443029403687412) ) ) {
                        result[0] += 0.01882668206078936;
                      } else {
                        result[0] += -0.10674573282675415;
                      }
                    } else {
                      result[0] += -0.05110111995608374;
                    }
                  }
                } else {
                  result[0] += -0.04366655826870317;
                }
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.02841608903965259;
              } else {
                result[0] += -0.21555427270584532;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.09051582329297908;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.005094666439371654;
                } else {
                  result[0] += 0.10936121684170551;
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.10418747915450917;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.02701340523814774;
                    } else {
                      result[0] += 0.07072521032186019;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.025649602338512607;
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.05627397149012593;
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.007803275844645217;
                          } else {
                            result[0] += -0.039099665641355495;
                          }
                        } else {
                          result[0] += -0.06971665466132086;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0027797492940589935;
                  }
                }
              }
            }
          } else {
            result[0] += -0.0577349350827898;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.004596426984857688;
          } else {
            result[0] += 0.0012956298893166325;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.07202001697090162;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += -0.004612379909714871;
                  } else {
                    result[0] += -0.03653234378946387;
                  }
                }
              } else {
                result[0] += -0.010343258712426279;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.847881793975830966) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.000367528333172036;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.529265403747559482) ) ) {
                        result[0] += -0.019192482102354292;
                      } else {
                        result[0] += -0.04982679752021393;
                      }
                    }
                  } else {
                    result[0] += -0.0014951834294479015;
                  }
                } else {
                  result[0] += 0.0021031593045634954;
                }
              } else {
                result[0] += -0.060577251116300204;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                result[0] += -0.00485091416533758;
              } else {
                result[0] += -0.03429546871743086;
              }
            } else {
              if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.011982314570462754;
                } else {
                  result[0] += 0.030270038355599;
                }
              } else {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    result[0] += -0.019845563481891853;
                  } else {
                    result[0] += -0.11743998423980645;
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0008325192397860598;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.0013121979728293706;
                        } else {
                          result[0] += 0.026721209871002485;
                        }
                      }
                    } else {
                      result[0] += -0.08369153973985387;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.006127195084242032;
                    } else {
                      result[0] += -0.13909890797555732;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += 0.016703511279188756;
        } else {
          result[0] += 0.13393880023441154;
        }
      }
    } else {
      result[0] += -0.030987578852120574;
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.914978742599488193) ) ) {
            result[0] += 0.02120457876939671;
          } else {
            result[0] += 0.002497703166627962;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
            result[0] += -0.04546363113009144;
          } else {
            result[0] += -0.00889001278838026;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
            result[0] += 0.030094907035935554;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
              result[0] += 0.03452107870244204;
            } else {
              result[0] += -0.03410917534454544;
            }
          }
        } else {
          result[0] += -0.03739762890267732;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.42895507812500178) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.05257330631113845;
                  } else {
                    result[0] += 0.09796800520410119;
                  }
                } else {
                  result[0] += 0.0011952768044379992;
                }
              } else {
                result[0] += 0.017212320875091447;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.04074122787250846;
                } else {
                  result[0] += -0.012636264664902029;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                  result[0] += -0.03264675194321292;
                } else {
                  result[0] += 0.05051738795529767;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.0036105399608910706;
              } else {
                result[0] += -0.02788382371734749;
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.004935823189907265;
              } else {
                result[0] += -0.023545210061580307;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.004029488556195983;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.028283002197330494;
                  } else {
                    result[0] += -0.06407610655193573;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.86220884323120206) ) ) {
                      result[0] += 0.015366382947978312;
                    } else {
                      result[0] += -0.09426720491921559;
                    }
                  } else {
                    result[0] += -0.022550288221896274;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                  result[0] += -0.01132960303105675;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      result[0] += 0.03151629601955328;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += -0.07741004458328644;
                      } else {
                        result[0] += 0.029745844417759598;
                      }
                    }
                  } else {
                    result[0] += -0.032039560640831626;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.09300499370396045;
              } else {
                result[0] += 0.0038369313409888153;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
            result[0] += 0.026050131649244697;
          } else {
            result[0] += -0.0052402397143065755;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
            result[0] += -0.0650483877630849;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.060643158587919244;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.0023694566456479134;
                  } else {
                    result[0] += -0.04163932680062282;
                  }
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.08007775151592078;
                  } else {
                    result[0] += 0.002196128368492877;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.303973913192749912) ) ) {
                result[0] += -0.08918095936496286;
              } else {
                result[0] += 0.005522455994899776;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                result[0] += -0.004775071997652546;
              } else {
                result[0] += -0.046198549567656115;
              }
            } else {
              result[0] += -0.03352235354341812;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                result[0] += 0.002321843901409228;
              } else {
                result[0] += -0.02158791870878096;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.338562726974488193) ) ) {
                  result[0] += 0.014021679505566424;
                } else {
                  result[0] += -0.03349930693417537;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.03886718667409997;
                } else {
                  result[0] += 0.00738325855250377;
                }
              }
            }
          }
        } else {
          result[0] += -0.07386413862395384;
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.0013860115627458804;
        } else {
          result[0] += -0.06774832535654898;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.01576483099995921;
          } else {
            result[0] += 0.0254249673533224;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.338562726974488193) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.874179124832154208) ) ) {
                result[0] += 0.017729279807957493;
              } else {
                result[0] += 0.050422436143908234;
              }
            } else {
              result[0] += 0.07771242987045308;
            }
          } else {
            result[0] += -0.03648498182443001;
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.029350534525689604;
        } else {
          result[0] += 0.021276551190689514;
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += 9.75397121132977e-06;
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.010202479638626034;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.031281265375688684;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0113354134691088;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += 0.08338114106626314;
                    } else {
                      result[0] += -0.09525982475561179;
                    }
                  } else {
                    result[0] += 0.027750274714391995;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                  result[0] += -0.07906014616982995;
                } else {
                  result[0] += 0.01673979633030673;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.668153762817383701) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.08457739860338809;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += -0.08143400758942178;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                  result[0] += -0.025294731735092408;
                } else {
                  result[0] += 0.024991821179341435;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.314458370208742011) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)5.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.009160084391470837;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                    result[0] += -0.021508142728622063;
                  } else {
                    result[0] += 0.05311785483708875;
                  }
                }
              } else {
                result[0] += -0.0025687613325272108;
              }
            } else {
              result[0] += 0.07409883206535778;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.05994848674871032;
            } else {
              result[0] += -0.006396395832760766;
            }
          } else {
            result[0] += -0.08000260907424701;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.0013989496638279078;
          } else {
            result[0] += -0.09536091838379937;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.497191667556763583) ) ) {
            result[0] += 0.0005502584221067393;
          } else {
            result[0] += 0.07035378296402102;
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                result[0] += -0.08360316475347666;
              } else {
                result[0] += -0.13578165523861915;
              }
            } else {
              result[0] += -0.024237729637259457;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.497191667556763583) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                  result[0] += -0.035552877413958704;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                    result[0] += -0.045373923026434226;
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.17534079722416943;
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                        result[0] += 0.009636428662668903;
                      } else {
                        result[0] += 0.05318416604001082;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                  result[0] += -0.06806712749922586;
                } else {
                  result[0] += 0.01925824082817468;
                }
              }
            } else {
              result[0] += 0.06564044851140605;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79285955429077326) ) ) {
            result[0] += -0.01017964060469299;
          } else {
            result[0] += -0.04766494846481212;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                result[0] += 0.0832059476930573;
              } else {
                result[0] += 0.018228847783804573;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.05783528486540627;
                } else {
                  result[0] += -0.10843200739446941;
                }
              } else {
                result[0] += 0.08630845629775057;
              }
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                  result[0] += 0.010682703567961416;
                } else {
                  result[0] += 0.13329303076769158;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                          result[0] += -0.07716705157183634;
                        } else {
                          result[0] += 0.0436468400061963;
                        }
                      } else {
                        result[0] += -0.06578705206920445;
                      }
                    } else {
                      result[0] += 0.031518431165823035;
                    }
                  } else {
                    result[0] += -0.1105220070192635;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.03579256711683537;
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.029528200385496285;
                    } else {
                      result[0] += 0.11092338507360555;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.874179124832154208) ) ) {
                result[0] += 0.016710067204407573;
              } else {
                result[0] += 0.05308242667136548;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.02246241223728295;
          } else {
            result[0] += 0.12856100061990305;
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.117535036977465;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
              result[0] += -0.07561948579838787;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.041386220689585515;
              } else {
                result[0] += 0.03574131406246441;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.12105351196282055;
        } else {
          result[0] += -0.055979142474785484;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.0010545322716531053;
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.008457748562097144;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
                    result[0] += -0.0010039103345502564;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
                      result[0] += -0.05824468609299632;
                    } else {
                      result[0] += 0.003655919663511162;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.11111859544723786;
                      } else {
                        result[0] += -0.013094727774888536;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.0033508047076857145;
                          } else {
                            result[0] += -0.17695506830937532;
                          }
                        } else {
                          result[0] += 0.00753424351923253;
                        }
                      } else {
                        result[0] += -0.020412660545515972;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                      result[0] += 0.00603948900268633;
                    } else {
                      result[0] += -0.009902813191316528;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.003838300704956943) ) ) {
                      result[0] += -0.003282724316122639;
                    } else {
                      result[0] += 0.00953399591889047;
                    }
                  } else {
                    result[0] += -0.011376716022564314;
                  }
                }
              }
            } else {
              result[0] += 0.006307630738845534;
            }
          } else {
            result[0] += 0.03196885886217243;
          }
        } else {
          result[0] += -0.004899676839799402;
        }
      } else {
        result[0] += 0.02884584218200283;
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.12873522829210163;
          } else {
            result[0] += -0.018468034782409697;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10166215896606623) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                  result[0] += 0.029445209363870017;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                    result[0] += 0.026505986922099368;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.02453641205900149;
                    } else {
                      result[0] += 0.025447488168425528;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  result[0] += 0.039741980903251525;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.04354849268536987;
                  } else {
                    result[0] += 0.007252216426815574;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.138696432113648349) ) ) {
                result[0] += -0.009943591057098184;
              } else {
                result[0] += -0.04449891583096353;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.0014380433518079222;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0005071547328013436;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.655387401580811435) ) ) {
                      result[0] += -0.004857963459365407;
                    } else {
                      result[0] += -0.03839040620548295;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.019841762287522043;
                    } else {
                      result[0] += -0.055633449396485846;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.001513339365358224;
                    } else {
                      result[0] += -0.03082663124665972;
                    }
                  } else {
                    result[0] += 0.006021815678172786;
                  }
                } else {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.013661579791611492;
                  } else {
                    result[0] += 0.0700983201500589;
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.04689344286797143;
                } else {
                  result[0] += -0.02706775828242143;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.01449309584250372;
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.004624297168758684;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                    result[0] += 0.040131420977198955;
                  } else {
                    result[0] += -0.011945260346646266;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.018646398553589252;
                  } else {
                    result[0] += 0.042799033287167686;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.305786132812500888) ) ) {
                    result[0] += 0.009812016521146012;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                        result[0] += -0.0386261595110613;
                      } else {
                        result[0] += 0.08739928728437613;
                      }
                    } else {
                      result[0] += -0.009932249173398584;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.02525897178620617;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                    result[0] += 0.001120463446745582;
                  } else {
                    result[0] += 0.03871682635697403;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.02998224646797773;
                  } else {
                    result[0] += -0.09745406233296688;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.00817812691392005;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.11107802215133654;
                    } else {
                      result[0] += -0.027565894815825137;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.11972198206169081;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37995386123657404) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.02478507280340697;
              } else {
                result[0] += 0.004538905913416824;
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.002783056799845073;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                  result[0] += 0.03627523875105148;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                    result[0] += -0.019470105410827013;
                  } else {
                    result[0] += -0.0670352459314472;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.02053275115147548;
            } else {
              result[0] += 0.002446513073333106;
            }
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.001871571490383606;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
                result[0] += 0.003381144925161474;
              } else {
                result[0] += -0.03903549098144274;
              }
            } else {
              result[0] += -0.0607252621157266;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            result[0] += -0.03483759556005538;
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.038877586616866744;
            } else {
              result[0] += -0.01424089301790803;
            }
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
                result[0] += -0.018282697040281327;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                        result[0] += -0.03882377021957542;
                      } else {
                        result[0] += 0.051374175579840045;
                      }
                    } else {
                      result[0] += -0.05390648301615423;
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.437634944915773261) ) ) {
                      result[0] += 0.02687689716166407;
                    } else {
                      result[0] += -0.01754812331816252;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)2.297559976577759233) ) ) {
                        result[0] += 0.05308719650676435;
                      } else {
                        result[0] += -0.1330897422951936;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.01062502318042974;
                      } else {
                        result[0] += -0.06064998275642969;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                        result[0] += 0.08696803307222167;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += -0.08105816493932061;
                        } else {
                          result[0] += -0.015668566960442977;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.297559976577759233) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.04512849715949506;
                          } else {
                            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                              result[0] += -0.10415521614961794;
                            } else {
                              result[0] += 0.002158305989974492;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.06977090242172304;
                          } else {
                            result[0] += -0.02219489056206892;
                          }
                        }
                      } else {
                        result[0] += -0.023501059594069534;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.06195117536771527;
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.03736172809609626;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.358708143234253818) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.038702220610693006;
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.73715209960937678) ) ) {
                          result[0] += 0.009893225776971238;
                        } else {
                          result[0] += -0.097060377284173;
                        }
                      }
                    } else {
                      result[0] += 0.07467195099323481;
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.012317331755465143;
                      } else {
                        result[0] += -0.061832589461582044;
                      }
                    } else {
                      result[0] += 0.03307555096724165;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.21116338250951772;
                    } else {
                      result[0] += 0.00606852220414746;
                    }
                  } else {
                    result[0] += -0.044303554002431;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  result[0] += 0.006055739185758781;
                } else {
                  result[0] += -0.03615317956293222;
                }
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.07541743770179726;
                  } else {
                    result[0] += -0.007188935090132746;
                  }
                } else {
                  result[0] += -0.004278271880346607;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
              result[0] += 0.00787788300603917;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                result[0] += 0.06901293350086855;
              } else {
                result[0] += -0.04135180418794571;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
              result[0] += -0.002986844697204301;
            } else {
              result[0] += -0.024637776585200027;
            }
          }
        } else {
          result[0] += 0.0009210535493238553;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.632926940917970526) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.308072090148926669) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
                result[0] += -0.004751312964423253;
              } else {
                result[0] += 0.047272925611041405;
              }
            } else {
              result[0] += 0.04490549406980807;
            }
          } else {
            result[0] += -0.015224618278913308;
          }
        } else {
          result[0] += -0.02007062804139588;
        }
      }
    }
  } else {
    result[0] += 0.0008469023862976183;
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += 0.09295527726434515;
              } else {
                result[0] += 0.009760834858572766;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.037794558548616214;
              } else {
                result[0] += 0.008453406389156218;
              }
            }
          } else {
            result[0] += -0.017574126421275635;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.080862283706665927) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += 0.010243846106672406;
                } else {
                  result[0] += -0.002561780031699382;
                }
              } else {
                result[0] += -0.016873354007308942;
              }
            } else {
              if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.03957214906515094;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += 0.05165757914956418;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += 0.0018320942763380714;
                    } else {
                      result[0] += -0.03567592551129846;
                    }
                  }
                }
              } else {
                result[0] += 0.00021020951385745292;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.5655555725097674) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += 0.019128310140590753;
                  } else {
                    result[0] += 0.0819539034825712;
                  }
                } else {
                  result[0] += -0.049094368354832635;
                }
              } else {
                result[0] += -0.05599635563057231;
              }
            } else {
              result[0] += 0.00032343829112054515;
            }
          }
        }
      } else {
        result[0] += 0.06778825564539742;
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
          result[0] += -0.02055981435289197;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.011621157855932531;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
                result[0] += 0.04511797148273047;
              } else {
                result[0] += -0.0791052332002454;
              }
            } else {
              result[0] += 0.10802018648793064;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.0018428367035534133;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                result[0] += 0.02152504136111157;
              } else {
                result[0] += 0.1040808726212563;
              }
            }
          } else {
            result[0] += -0.008082104398057567;
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.006910842257622918;
            } else {
              result[0] += -0.06097419965856669;
            }
          } else {
            result[0] += -0.018268244848012875;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
        result[0] += -0.008121720185790807;
      } else {
        result[0] += -0.047161309959428216;
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.589327573776246005) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.00015709801835633419;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                  result[0] += -0.022858685993452393;
                } else {
                  result[0] += 0.0021189737075493886;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40695333480835139) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                    result[0] += 0.006176451602750548;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                      result[0] += 0.0038090394138002985;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                          result[0] += -0.0790888495358158;
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                            result[0] += -0.023884676935241975;
                          } else {
                            result[0] += 0.00027399240712720546;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.030821955431225484;
                        } else {
                          result[0] += 0.010359060494420987;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.012052040310733982;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                  result[0] += 0.022742766352154845;
                } else {
                  result[0] += -0.013079200496086456;
                }
              }
            }
          } else {
            result[0] += -0.010936077275793571;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.497191667556763583) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.07099983115444607;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.057527091924188194;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.869292974472046787) ) ) {
                        result[0] += 0.009480406482680255;
                      } else {
                        result[0] += -0.030161040909346327;
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.005463224346370989;
                      } else {
                        result[0] += 0.021506154990561046;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += -0.10406313719733998;
                    } else {
                      result[0] += -0.01319131989981915;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.382196187973023349) ) ) {
                      result[0] += -0.0071157262189952;
                    } else {
                      result[0] += 0.005803943940282045;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += -0.010567389170621437;
                  } else {
                    result[0] += 0.01510675751541921;
                  }
                } else {
                  result[0] += -0.04375086133798892;
                }
              }
            }
          } else {
            result[0] += 0.023023706482683277;
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          result[0] += 0.013371970092327332;
        } else {
          result[0] += -0.05048214502725752;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.024128588275892496;
          } else {
            result[0] += -0.019946939879054725;
          }
        } else {
          result[0] += -0.016794776207796916;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
              result[0] += 0.010142292813818246;
            } else {
              result[0] += 0.08336121197241032;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
              result[0] += 0.017451408782018337;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                result[0] += 0.13365803594981113;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                  result[0] += 0.0003635274331228924;
                } else {
                  result[0] += -0.05267301547516809;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.725620865821838823) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                  result[0] += 0.008057040723538687;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += 0.0063133827139775335;
                  } else {
                    result[0] += 0.07493103240720181;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      result[0] += -0.005600665533049918;
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.025361476339936708;
                      } else {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += 0.16206953586073158;
                        } else {
                          result[0] += 0.03571750214901482;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.861792564392090288) ) ) {
                      result[0] += 0.03623608802838514;
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.010769641906219669;
                        } else {
                          result[0] += -0.13651864721595366;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                          result[0] += 0.14288076953331771;
                        } else {
                          result[0] += -0.10928223918014318;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.013644383236789993;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.003574266368420002;
              } else {
                result[0] += 0.05630813338311108;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.01589310056722859;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.060294389724732333) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.95005559921264826) ) ) {
                          result[0] += -0.07401618737126998;
                        } else {
                          result[0] += 0.07056888387422243;
                        }
                      } else {
                        result[0] += 0.09596245437920176;
                      }
                    } else {
                      result[0] += -0.0017153827981769153;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += -0.11345095400541594;
                    } else {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += -0.010585816188728435;
                      } else {
                        result[0] += 0.029640329492809897;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.004655914401331162;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.05967275267106109;
                    } else {
                      result[0] += 0.060326021797534816;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.832297801971436435) ) ) {
                  result[0] += 0.026883298383278315;
                } else {
                  result[0] += -0.003151326431078825;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.0014642965607662628;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.46093606948852717) ) ) {
              result[0] += 0.04848091027994186;
            } else {
              result[0] += -0.06454934458204584;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              result[0] += -0.03265210227811476;
            } else {
              result[0] += 0.0907932762061402;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.027851475506854764;
          } else {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.011131943346508855;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                  result[0] += -0.13228407435981457;
                } else {
                  result[0] += 0.016990164552293335;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.02031139225982086;
                } else {
                  result[0] += 0.007130406330804799;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.022013331581023112;
                } else {
                  if ( UNLIKELY(  (data[35].missing != -1) && (data[35].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.12566139810934027;
                  } else {
                    result[0] += -0.06427704930457598;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.015114222496975313;
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06785045143004617;
              } else {
                result[0] += -0.013898368773065765;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.57691621780395685) ) ) {
                result[0] += -0.08409877966344231;
              } else {
                result[0] += 0.07924943704762376;
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03814915351338111;
              } else {
                result[0] += -0.020815720142241097;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.18732333183288663) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.03258433465748314;
        } else {
          result[0] += -0.03195846792858129;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
          result[0] += -0.006197620395254922;
        } else {
          result[0] += -0.04824743611574856;
        }
      }
    } else {
      result[0] += 0.0004510525736409953;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
              result[0] += 0.0791042841113756;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.03220876474195201;
              } else {
                result[0] += -0.06622457000844781;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.03745147091777383;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.01538421005708088;
              } else {
                result[0] += -0.03614527665467151;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
              result[0] += 0.02353157273189275;
            } else {
              result[0] += -0.04853635896892611;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.004655247287463359;
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                  result[0] += -0.031125301846615074;
                } else {
                  result[0] += -0.10745130080377088;
                }
              } else {
                result[0] += -0.00513128192417109;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.016639371512196983;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.986227512359620917) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.080862283706665927) ) ) {
                      result[0] += 0.006279476931495604;
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.027993581101615036;
                        } else {
                          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.1182657568089439;
                          } else {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.531409263610840732) ) ) {
                              result[0] += 0.0374161774175253;
                            } else {
                              result[0] += -0.03665221348040221;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.0306114508101958;
                      }
                    }
                  } else {
                    result[0] += -0.004529136737473774;
                  }
                } else {
                  result[0] += 0.01614896064534085;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.03710758182120184;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                    result[0] += -0.008782698709447105;
                  } else {
                    result[0] += 0.017776834952367107;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                  result[0] += 0.04047921749902147;
                } else {
                  result[0] += -0.050100929998101784;
                }
              } else {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.010842350489678337;
                    } else {
                      result[0] += 0.015031979534015284;
                    }
                  } else {
                    result[0] += -0.023426571697243645;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.01083376670048691;
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.048421062125799975;
                    } else {
                      result[0] += 0.002586123126632092;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
              result[0] += 0.08903449660989081;
            } else {
              result[0] += -0.007139964507773051;
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0353480024440792;
            } else {
              result[0] += 0.0004901730515184351;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.0012419566731331063;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            result[0] += -0.010404082771488344;
          } else {
            result[0] += 0.06571503672255452;
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.022763426965521787;
            } else {
              result[0] += -0.10139273148373973;
            }
          } else {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.009036219335420167;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.01711558226533788;
                } else {
                  result[0] += 0.004972772184521659;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.1212880857148085;
                  } else {
                    result[0] += -0.05946202490832084;
                  }
                } else {
                  result[0] += -0.01947030132477786;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.14105346099362898;
                } else {
                  result[0] += -0.07942843200805667;
                }
              } else {
                result[0] += 0.04750868020008081;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.022225309097943246;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.07437724510141826;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.019073436558048325;
                  } else {
                    result[0] += -0.08315837879000908;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.05324798808695497;
                  } else {
                    result[0] += 0.01647511642290425;
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.024388810721071835;
                      } else {
                        result[0] += -0.10151133139238777;
                      }
                    } else {
                      result[0] += -0.07556057269991254;
                    }
                  } else {
                    result[0] += 0.02543229309730003;
                  }
                }
              } else {
                result[0] += -0.09769379435527298;
              }
            } else {
              result[0] += -0.0004640640584993622;
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.00037419323965116554;
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
          result[0] += 0.05639303867692444;
        } else {
          result[0] += 0.007461440299962766;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.012068777079083357;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.725620865821838823) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                  result[0] += 0.05653629184475585;
                } else {
                  result[0] += 0.014460045224427327;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.017248630774398192;
                } else {
                  result[0] += -0.011184209438757496;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.0031366527919753657;
              } else {
                result[0] += 0.05098927464803138;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
              result[0] += 0.01092091434906388;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.10048965206369896;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += 0.04659920542254955;
                  } else {
                    result[0] += -0.0024577930805344357;
                  }
                }
              } else {
                result[0] += 0.05833891844345277;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.189540147781372958) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.428993701934816229) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                    result[0] += -0.03907507543543894;
                  } else {
                    result[0] += 0.14840891521707797;
                  }
                } else {
                  result[0] += -0.004506271544341569;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.05096810031257751;
                } else {
                  result[0] += 0.02206291477624757;
                }
              }
            } else {
              if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.029092020437827167;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += 0.020102392667905148;
                  } else {
                    result[0] += 0.0855371843646446;
                  }
                }
              } else {
                result[0] += -0.00642362023250688;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.0011263268925934184;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
              result[0] += 0.031853821691879246;
            } else {
              result[0] += -0.05657317339862616;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.343781709671021396) ) ) {
                result[0] += -0.045197294301409854;
              } else {
                result[0] += 0.08697441525207346;
              }
            } else {
              result[0] += 0.09270544271485262;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0255915446963372;
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                    result[0] += -0.0068394094130942405;
                  } else {
                    result[0] += 0.05343456633399682;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                    result[0] += 0.0052642013847114;
                  } else {
                    result[0] += -0.03065123485008363;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.0070378564169205965;
                    } else {
                      result[0] += -0.07126878714206666;
                    }
                  } else {
                    result[0] += 0.01612205762434926;
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.02446076771334677;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.049575957338205245;
                      } else {
                        result[0] += -0.016752443386838147;
                      }
                    } else {
                      result[0] += -0.08498202987500734;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.616744756698609287) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                result[0] += -0.050797803692241655;
              } else {
                result[0] += 0.08273397241115735;
              }
            } else {
              result[0] += -0.10526785290490287;
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += -0.05477198392930863;
          } else {
            result[0] += -0.021488456589101722;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.18732333183288663) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.02944965784594581;
          } else {
            result[0] += -0.028606565620504277;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
            result[0] += 0.005366560111892914;
          } else {
            result[0] += -0.03893880782538848;
          }
        }
      } else {
        if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.01358469424362121;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
              result[0] += -0.021643476205698522;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.0028086069860501646;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.02248754013418434;
                  } else {
                    result[0] += -0.0011916802072416256;
                  }
                }
              } else {
                result[0] += 0.009369840493061023;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
            result[0] += 0.0010752938925640103;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.045120265958328847;
                } else {
                  result[0] += -0.042528897556927486;
                }
              } else {
                result[0] += -0.021115361296549805;
              }
            } else {
              result[0] += -0.0011944081796980812;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        result[0] += 0.012907207260904777;
      } else {
        result[0] += -0.04756083064456078;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.000990371594952693;
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.0013766977924471585;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.587220668792725498) ) ) {
          result[0] += 0.001863485646257081;
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.0535190668968149;
          } else {
            result[0] += -0.03176966217596753;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.12496194759701401;
          } else {
            result[0] += -0.011961879590976776;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.003374807393103757;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
                  result[0] += 0.0032406566575515004;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
                      result[0] += 0.03326893882310323;
                    } else {
                      result[0] += 0.08523334557902618;
                    }
                  } else {
                    result[0] += -0.0060625530313299985;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                result[0] += -0.0033625512833025833;
              } else {
                result[0] += -0.03284481838017728;
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.655387401580811435) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.0004785067025425278;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.004199128857124432;
                    } else {
                      result[0] += -0.05391801836283744;
                    }
                  } else {
                    result[0] += 0.01957911170788558;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.023723868284245577;
                        } else {
                          result[0] += -0.039369426347932925;
                        }
                      } else {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.008340636263360734;
                        } else {
                          result[0] += 0.048010501576911675;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                        result[0] += -0.0011577401579994952;
                      } else {
                        result[0] += -0.03041886364445956;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.055571448698606835;
                    } else {
                      result[0] += 0.01508973212974005;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.07674845075694872;
                    } else {
                      result[0] += -0.02939204291218045;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                        result[0] += -0.04527101177586358;
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                            result[0] += 0.008680356310643865;
                          } else {
                            result[0] += -0.07203977438582747;
                          }
                        } else {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.010244862228483117;
                          } else {
                            result[0] += -0.05735702919005834;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += 0.007393642523840553;
                          } else {
                            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += 0.009883332511556576;
                            } else {
                              result[0] += 0.11348332655610904;
                            }
                          }
                        } else {
                          result[0] += 0.07435152400062132;
                        }
                      } else {
                        result[0] += -0.048441652568927906;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.01228469372098534;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.00028499469666778195;
            } else {
              result[0] += -0.01820944267627941;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.011023881828921897;
            } else {
              result[0] += -0.07110262740638339;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.586156606674195224) ) ) {
                  result[0] += -0.0022915536275681637;
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.048308770386663584;
                    } else {
                      result[0] += 0.016492841495113028;
                    }
                  } else {
                    result[0] += -0.019633884175073807;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                  result[0] += 0.026746743648067187;
                } else {
                  result[0] += -0.04315830461110704;
                }
              }
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.008600874693905955;
                } else {
                  result[0] += 0.03669367701936722;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.305786132812500888) ) ) {
                  result[0] += 0.010334661793254353;
                } else {
                  result[0] += 0.05980015064884098;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += 0.000991851521092543;
                } else {
                  result[0] += 0.03687310594260773;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.024012128745989417;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.005477297514204658;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.10048033682051626;
                    } else {
                      result[0] += -0.017718546215796396;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                  result[0] += -0.0714937421979112;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
                    result[0] += -0.020206129348029903;
                  } else {
                    result[0] += 0.050720808205950874;
                  }
                }
              } else {
                result[0] += -0.07604404177450097;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.015748422487031063;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.005694411544805696;
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.547126770019532138) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.988812565803528276) ) ) {
                result[0] += -0.05669196798817239;
              } else {
                result[0] += -0.010783427627813409;
              }
            } else {
              result[0] += -0.005763325100068505;
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.499747991561890537) ) ) {
                    result[0] += 0.046534650554440865;
                  } else {
                    result[0] += -0.011489711808666437;
                  }
                } else {
                  result[0] += -0.09856583221793323;
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.04925700502308379;
                } else {
                  result[0] += 0.06148017184978536;
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -2.4728342307080963e-05;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.00010363516532164593;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += 0.0038168827780085883;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                        result[0] += 0.014635407783516775;
                      } else {
                        result[0] += 0.061411651142096194;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.033564014661806196;
                  } else {
                    result[0] += -0.0549842044536975;
                  }
                } else {
                  result[0] += -0.07810579466553814;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.07951431061035596;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += 0.020294413404473754;
                } else {
                  result[0] += -0.08080097875065333;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.025963509139856858;
                  } else {
                    result[0] += -0.0856990056342962;
                  }
                } else {
                  result[0] += 0.053213788784730015;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.025907292887472788;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.035800519370181944;
                } else {
                  result[0] += 0.005604351032338261;
                }
              }
            } else {
              result[0] += 0.029465237344020387;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
                result[0] += 0.06523109424800697;
              } else {
                result[0] += 0.0014920873777136546;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.0027137934982213916;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.007749425324637894;
                      } else {
                        result[0] += -0.02279623282772122;
                      }
                    } else {
                      result[0] += -0.08688517775347075;
                    }
                  } else {
                    result[0] += 0.05512130021915765;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.08221258120442429;
                  } else {
                    result[0] += -0.02226762950881769;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.05307498253483112;
              } else {
                result[0] += -0.016591755488570153;
              }
            } else {
              result[0] += -0.004322727968825098;
            }
          } else {
            result[0] += 0.006150474150738121;
          }
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.500000000000000888) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.0008343504700972907;
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                      result[0] += 0.05656843401663454;
                    } else {
                      result[0] += -0.019627795635685966;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.027917689605584224;
                    } else {
                      result[0] += -0.10407369847135511;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.06469820934940088;
                    } else {
                      result[0] += 0.04885520298375421;
                    }
                  } else {
                    result[0] += -0.1055042572916344;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.03196493272831478;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += -0.023448787617273904;
                    } else {
                      result[0] += 0.12574794218426488;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.04602239063012404;
                    } else {
                      result[0] += -0.05130159466998964;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.028201103210450107) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.041140299211562634;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.00796612610963272;
                  } else {
                    result[0] += 0.0363668725892293;
                  }
                }
              } else {
                result[0] += -0.0186908490342252;
              }
            } else {
              result[0] += 0.06620951108742529;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.01834637169495291;
            } else {
              result[0] += -0.03421674127211116;
            }
          } else {
            result[0] += -0.002869593044022902;
          }
        }
      }
    }
  } else {
    result[0] += 0.0006267630220037197;
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.014983822499403932;
  } else {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.00031261267003190754;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.497191667556763583) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.11326837539672896) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.048820021709569304;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.4822273254394549) ) ) {
                      result[0] += -0.01616060331555623;
                    } else {
                      result[0] += 0.007896816188230635;
                    }
                  }
                } else {
                  result[0] += -0.05711352335026786;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.08169558598045751;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.767332553863526279) ) ) {
                        result[0] += 0.006424349206494182;
                      } else {
                        result[0] += -0.0235912336631216;
                      }
                    }
                  } else {
                    result[0] += -0.07821855481174045;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.01700892383432689;
                        } else {
                          result[0] += -0.03791122415701014;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.670753479003907138) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.726826429367066318) ) ) {
                              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                                result[0] += -0.026225591944293154;
                              } else {
                                result[0] += 0.0277968130330295;
                              }
                            } else {
                              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.037648494431911365;
                              } else {
                                result[0] += 0.007570582021763345;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
                                  result[0] += 0.049772009341225706;
                                } else {
                                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                                    result[0] += 0.029143553993871103;
                                  } else {
                                    result[0] += -0.06340733445230634;
                                  }
                                }
                              } else {
                                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                                    result[0] += 0.0055471911127549485;
                                  } else {
                                    result[0] += 0.07526524771143185;
                                  }
                                } else {
                                  result[0] += -0.07666785854914025;
                                }
                              }
                            } else {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += -0.010457102464223975;
                                } else {
                                  result[0] += -0.11219639927524888;
                                }
                              } else {
                                result[0] += 0.02922376285186424;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.01150435542721754;
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                              result[0] += 0.040804481924267784;
                            } else {
                              result[0] += 0.08405990122657671;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
                                result[0] += 0.022604575673026078;
                              } else {
                                result[0] += -0.016750374175814424;
                              }
                            } else {
                              result[0] += 0.03319185670380168;
                            }
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.980170249938965732) ) ) {
                                result[0] += -0.09389895799419677;
                              } else {
                                result[0] += -0.14941487867010297;
                              }
                            } else {
                              result[0] += 0.00365816377498521;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.947025299072267401) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.297764539718628818) ) ) {
                              result[0] += -0.039921960266492834;
                            } else {
                              result[0] += 0.050302651544108956;
                            }
                          } else {
                            result[0] += 0.023504554590956444;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.329718828201294833) ) ) {
                                result[0] += 0.011548658627578624;
                              } else {
                                result[0] += -0.10406894530327303;
                              }
                            } else {
                              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                                  result[0] += -0.01924424877544929;
                                } else {
                                  result[0] += 0.05696844084535027;
                                }
                              } else {
                                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.350257158279419833) ) ) {
                                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                                    result[0] += -0.004278090057204753;
                                  } else {
                                    result[0] += 0.07225968628780104;
                                  }
                                } else {
                                  result[0] += 0.1276270818134352;
                                }
                              }
                            }
                          } else {
                            result[0] += -0.027097352569540645;
                          }
                        } else {
                          result[0] += -0.11962565831582449;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.05856742375008515;
                          } else {
                            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += 0.10633853338893239;
                            } else {
                              result[0] += 0.03345500201007681;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                            result[0] += -0.08952457386112334;
                          } else {
                            result[0] += -0.014218151335304817;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -8.366335299776214e-05;
                        } else {
                          result[0] += -0.07791616740170944;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += 0.051428594994238234;
                      } else {
                        result[0] += -0.024843362265699147;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.06742638975824622;
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                result[0] += -0.04417045723022401;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.08954543580850749;
                } else {
                  result[0] += 0.04412187577567423;
                }
              }
            } else {
              result[0] += 0.0073731015103273055;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += 0.06027638355664813;
        } else {
          result[0] += 0.007754295614644888;
        }
      }
    } else {
      result[0] += -0.02720455231727726;
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += -0.00027341646480529496;
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.497191667556763583) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.043645281953411025;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.4822273254394549) ) ) {
              result[0] += -0.015279358026607702;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                result[0] += -0.0578528795911599;
              } else {
                result[0] += 0.011376369052719376;
              }
            }
          }
        } else {
          result[0] += -0.05297533918369873;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.07634783810984531;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.670753479003907138) ) ) {
                result[0] += 0.007289737337766457;
              } else {
                result[0] += -0.018572338473498564;
              }
            }
          } else {
            result[0] += -0.07488191252101686;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.970085620880127397) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.539549827575684482) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80505037307739435) ) ) {
                      result[0] += -0.01918039441899623;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                          result[0] += -0.010962414212418316;
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.06835034492552858;
                          } else {
                            result[0] += -0.02389081934973919;
                          }
                        }
                      } else {
                        result[0] += 0.032199105892226944;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.0711120014734331;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.025133483794958784;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                            result[0] += -0.06934509292409276;
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += 0.010496564243651566;
                            } else {
                              result[0] += -0.09534540162999579;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                            result[0] += 0.07525004842386991;
                          } else {
                            result[0] += -0.03774139989424499;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                              result[0] += 0.018393771258046723;
                            } else {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                                result[0] += -0.096210259308487;
                              } else {
                                result[0] += 0.021093557369114398;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                                  result[0] += 0.06015936786696374;
                                } else {
                                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                                      result[0] += 0.09375347911531873;
                                    } else {
                                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                                        result[0] += -0.06876928532044778;
                                      } else {
                                        result[0] += 0.12275746612069;
                                      }
                                    }
                                  } else {
                                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                                        result[0] += -0.0023700653590301903;
                                      } else {
                                        result[0] += -0.1300159591877713;
                                      }
                                    } else {
                                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                                        result[0] += 0.07774229042920976;
                                      } else {
                                        result[0] += -0.031210439293262622;
                                      }
                                    }
                                  }
                                }
                              } else {
                                result[0] += 0.07967757451684115;
                              }
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                                result[0] += 0.028482038430252706;
                              } else {
                                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                                  result[0] += -0.09172910170882385;
                                } else {
                                  result[0] += 0.018971203442989085;
                                }
                              }
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += -0.05812320153055185;
                        } else {
                          result[0] += 0.003829072592871577;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.66339445114135831) ) ) {
                      result[0] += -0.0035195412467199706;
                    } else {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.02877459142709579;
                      } else {
                        result[0] += 0.07130033188099745;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                      result[0] += -0.05090050725798978;
                    } else {
                      result[0] += 0.01221718103339067;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.05039491971015534;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += 0.10369430627793895;
                        } else {
                          result[0] += 0.03664022095699313;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)7.496087312698365146) ) ) {
                        result[0] += -0.06490113304291505;
                      } else {
                        result[0] += 0.019934465020520754;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += -0.08597290651059508;
                    } else {
                      result[0] += -0.015379944821133149;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.041385852692654365;
                  } else {
                    result[0] += 0.04115247987745652;
                  }
                }
              }
            } else {
              result[0] += 0.04111884582904922;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
              result[0] += -0.07885258942577641;
            } else {
              result[0] += 0.012093621924705839;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
          result[0] += 0.02426061468880708;
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.1410438258723747;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.11307696008315951;
              } else {
                result[0] += 0.04092788547627382;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.12433708679665588;
            } else {
              result[0] += 0.05791463510984795;
            }
          }
        }
      } else {
        result[0] += 0.00784200076210212;
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      result[0] += -0.00042222902180940404;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.395718574523926669) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
            result[0] += -0.0069633292688339975;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.02227889141920177;
            } else {
              result[0] += 0.04951312970523808;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            result[0] += 0.020664142029960815;
          } else {
            result[0] += -0.07989977464539186;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
            result[0] += 0.01518465900308201;
          } else {
            result[0] += -0.05257000375323643;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.1261270575715537;
            } else {
              result[0] += -0.04480690243753618;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.797939777374268466) ) ) {
                result[0] += 0.07992165200372929;
              } else {
                result[0] += -0.009397330171086314;
              }
            } else {
              result[0] += -0.018143027638212458;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.03197337335449172;
          } else {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              result[0] += -0.029047716678161667;
            } else {
              result[0] += 0.0033694983984923622;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.07782760452950913;
            } else {
              result[0] += -0.004182515691020772;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.020127415657043901) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                  result[0] += -0.0021420706551261983;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.02385884078957836;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.052870441740759194;
                    } else {
                      result[0] += 0.03226693133041844;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
                    result[0] += -0.03731909994489542;
                  } else {
                    result[0] += 0.0301436185379056;
                  }
                } else {
                  result[0] += -0.0009258824761824808;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.07807694654890511;
              } else {
                result[0] += 0.01824884969715334;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12460899353027521) ) ) {
          result[0] += -0.06598119555513798;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.034114168570356145;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.04220693634837605;
                } else {
                  result[0] += -0.06968574324904067;
                }
              } else {
                result[0] += -0.027419002062818543;
              }
            }
          } else {
            result[0] += -0.05562032208920942;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -0.03761826128003215;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.556798219680787021) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.010886714300488463;
          } else {
            result[0] += 0.015042964084192496;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += -0.035309237092083605;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.920236110687256748) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                        result[0] += 0.05374713369191342;
                      } else {
                        result[0] += 0.009619839576896272;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.827628135681153232) ) ) {
                        result[0] += 0.06685033526688593;
                      } else {
                        result[0] += 0.13081715582433515;
                      }
                    }
                  } else {
                    result[0] += 0.012470463317534717;
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.08797481354084566;
                  } else {
                    result[0] += -0.008369542783668136;
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.998158693313599077) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.518026351928711826) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                          result[0] += -0.03822823518884802;
                        } else {
                          result[0] += 0.03484716073043092;
                        }
                      } else {
                        result[0] += 0.06732633673608178;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.002004305601216027;
                        } else {
                          result[0] += -0.09771114658279939;
                        }
                      } else {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += 0.03198865465048852;
                        } else {
                          result[0] += -0.12281245351451915;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.03495136929670733;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                      result[0] += -0.07471897688940063;
                    } else {
                      result[0] += 0.03408956737412855;
                    }
                  } else {
                    result[0] += -0.09161525112488089;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.12631074149737265;
              } else {
                result[0] += -0.04614999189382242;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.870205879211427558) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.134879350662232333) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                      result[0] += -0.07310088482390344;
                    } else {
                      result[0] += 0.04768955562382937;
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.021278834840926162;
                    } else {
                      result[0] += 0.10340998616052742;
                    }
                  }
                } else {
                  result[0] += 0.13083195810328205;
                }
              } else {
                result[0] += 0.007099516663041008;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.01848646989480184;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.463808774948121005) ) ) {
                result[0] += 0.022152753719181868;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  result[0] += 0.02091861650924419;
                } else {
                  result[0] += -0.03136461724344549;
                }
              }
            } else {
              result[0] += -0.004629129503391592;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.002803992598040462;
            } else {
              result[0] += -0.03383427402516292;
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.966960191726685458) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += -0.0026088211995182824;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                  result[0] += -0.016226890748160763;
                } else {
                  result[0] += -0.06558451537636374;
                }
              }
            } else {
              result[0] += -0.04964777805779892;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.03248356550496599;
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.003621788118721622;
                } else {
                  result[0] += -0.06521385595798655;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.109245061874390537) ) ) {
                  result[0] += 0.03524424563383519;
                } else {
                  result[0] += -0.010893104929142847;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.035496537743551236;
            } else {
              result[0] += 0.004981991436001312;
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += -0.01867494767983406;
          } else {
            result[0] += -0.05407079620928856;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
          result[0] += 0.01435738965066877;
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.004568833528141634;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.0001578136552501383;
                } else {
                  result[0] += -0.04109433919491834;
                }
              } else {
                result[0] += -0.10176560458324456;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.05769836226925434;
              } else {
                result[0] += -0.02040940966710332;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
                  result[0] += -0.002402097096538037;
                } else {
                  result[0] += -0.07639718437169983;
                }
              } else {
                result[0] += -0.03215112434705835;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += -0.013139170889946414;
        } else {
          result[0] += 0.0014183198335626765;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.308072090148926669) ) ) {
          result[0] += -0.0015637228699155164;
        } else {
          result[0] += -0.048190032727560846;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.309873342514038974) ) ) {
              result[0] += 0.0007994244700878617;
            } else {
              result[0] += -0.0435040465527855;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.531007289886475498) ) ) {
                  result[0] += 0.024020662810964433;
                } else {
                  result[0] += 0.060995411312648476;
                }
              } else {
                result[0] += 0.05959160882652334;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.05839186907918878;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.837148427963257724) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.449861526489258257) ) ) {
                    result[0] += 0.0025492927243091165;
                  } else {
                    result[0] += -0.07811749597210343;
                  }
                } else {
                  result[0] += -0.11974581117202412;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.040232470487496744;
            } else {
              result[0] += -0.006587251055715307;
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.023400783538819248) ) ) {
              result[0] += 0.04164153929121511;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                result[0] += 0.11853729949620234;
              } else {
                result[0] += 0.03456149543645145;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
            result[0] += 0.02957748391287668;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
              result[0] += 0.031292921235977156;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.017591149596131346;
              } else {
                result[0] += 0.05167863512329667;
              }
            }
          }
        } else {
          result[0] += -0.04991200903498649;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.09361997012534795;
            } else {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.1052367809921157;
              } else {
                result[0] += -0.03516775190085052;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                result[0] += -0.0005025973971312683;
              } else {
                result[0] += -0.038018418399366864;
              }
            } else {
              result[0] += -0.052067610168237924;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.797939777374268466) ) ) {
              result[0] += 0.07145251356464986;
            } else {
              result[0] += -0.010080352184083203;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
              result[0] += 0.007177877573410441;
            } else {
              result[0] += -0.036804534560300396;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.32014131546020685) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              result[0] += 0.009666060159670533;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.009593325827402261;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.007406566769545291;
                } else {
                  result[0] += -0.03878954656886567;
                }
              }
            }
          } else {
            result[0] += 0.027921993746999236;
          }
        } else {
          result[0] += -0.017475675460193584;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += 0.010345542674906597;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.338562726974488193) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                    result[0] += -0.013492472224666058;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.09181462744291485;
                    } else {
                      result[0] += 0.021232788600616562;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.039055251657363406;
                  } else {
                    result[0] += -0.005914788328065175;
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.07517892795588432;
                } else {
                  result[0] += 0.008842906915426439;
                }
              }
            }
          } else {
            result[0] += 0.009921771943322732;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.03598846305802777;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.556798219680787021) ) ) {
                result[0] += -0.02716006683626676;
              } else {
                result[0] += -0.05866155366595908;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
              result[0] += 0.011165734405880395;
            } else {
              result[0] += -0.019841676089647396;
            }
          }
        }
      }
    } else {
      result[0] += -0.0006864616829054419;
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.99033999443054288) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
            result[0] += -0.010302664427591566;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.09008146402084796;
              } else {
                result[0] += 0.0021257372736708862;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.01951176821072063;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                  result[0] += 0.062374731494938024;
                } else {
                  result[0] += 0.020831658805301342;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.11149043118755181;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.03156609176042371;
              } else {
                result[0] += 0.01408312165110677;
              }
            } else {
              result[0] += -0.015168449857815744;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.59645986557007014) ) ) {
          result[0] += -0.06958389262699811;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
            result[0] += -0.046317356809331876;
          } else {
            result[0] += 0.017719558628180917;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.03477634394197867;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.845905780792238104) ) ) {
            result[0] += -0.014828285431506567;
          } else {
            result[0] += -0.05578903704428521;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.809862852096558505) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.020127415657043901) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
              result[0] += -0.006450647636167813;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.08992210227320396;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.568724632263184482) ) ) {
                      result[0] += 0.025695628122410816;
                    } else {
                      result[0] += -0.01374683916910948;
                    }
                  } else {
                    result[0] += -0.08426479303829906;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.966960191726685458) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.680079460144043857) ) ) {
                        result[0] += -0.008178183123064174;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.497191667556763583) ) ) {
                            result[0] += 0.014729941914196038;
                          } else {
                            result[0] += 0.07882729202462292;
                          }
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.76642942428588956) ) ) {
                              result[0] += 0.03496582203849477;
                            } else {
                              result[0] += -0.055129369663095965;
                            }
                          } else {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.04039056758932254;
                              } else {
                                result[0] += 0.015329218525107597;
                              }
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                                result[0] += -0.0072019711409748305;
                              } else {
                                result[0] += 0.023951457925070058;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                        result[0] += -0.07738802458988925;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
                            result[0] += -0.05360400039213105;
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.283562898635865146) ) ) {
                              result[0] += 0.03441760006321924;
                            } else {
                              result[0] += 0.11980420117925708;
                            }
                          }
                        } else {
                          result[0] += -0.01834475594177178;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                      result[0] += 0.028261431189344457;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80124759674072443) ) ) {
                        result[0] += -0.04528808161483844;
                      } else {
                        result[0] += 0.016071837377043773;
                      }
                    }
                  }
                } else {
                  result[0] += 0.03374534958958331;
                }
              }
            }
          } else {
            result[0] += -0.041071387992534986;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.11713556560085436;
          } else {
            result[0] += 0.05830552748917244;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
        if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            result[0] += 0.04288548067704953;
          } else {
            result[0] += 0.005439182831164423;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.03127502269269335;
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.016642527318726025;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.033664040269456956;
                } else {
                  result[0] += 0.005410758118226172;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.0141382636097392;
                  } else {
                    result[0] += 0.006130651854145171;
                  }
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.009277209243207121;
                    } else {
                      result[0] += -0.026633203784447907;
                    }
                  } else {
                    result[0] += -0.03018818623102082;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
            result[0] += -0.03184476489602627;
          } else {
            result[0] += -0.004400672133811084;
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                result[0] += 0.037309657719024614;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += 0.018654810627937692;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                      result[0] += 0.04290999129004211;
                    } else {
                      result[0] += -0.03910971083027759;
                    }
                  } else {
                    result[0] += -0.0484029970564695;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
                result[0] += -0.019484318276578236;
              } else {
                result[0] += 0.03323172388722117;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.002888161257143973;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.019650150657514406;
              } else {
                result[0] += -0.06564832046297978;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.0007462121133565806;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.0014237818198074457;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.0053802415599921724;
                } else {
                  result[0] += -0.0749268947797536;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                result[0] += -0.000816086402409029;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.020547187139788783;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.002047287791558369;
                  } else {
                    result[0] += -0.0310146456021498;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.05758719993351649;
            } else {
              result[0] += -0.009350462202391866;
            }
          }
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.0024922057632588185;
          } else {
            result[0] += -0.04414818162169709;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.917405366897583452) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.469231128692627841) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
                result[0] += -0.005113977797879947;
              } else {
                result[0] += -0.051620277853955725;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.86655306816101163) ) ) {
                  result[0] += 0.019552226624697272;
                } else {
                  result[0] += 0.07059188029799118;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
                  result[0] += 0.006217277175788012;
                } else {
                  result[0] += -0.06725163398000143;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.378218650817871982) ) ) {
                result[0] += 0.044078843221316305;
              } else {
                result[0] += 0.11333606861855126;
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0017077442296208685;
                } else {
                  result[0] += -0.061099771672076045;
                }
              } else {
                result[0] += 0.018621125110252365;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
                result[0] += -0.001368156237360943;
              } else {
                result[0] += 0.039673999511191646;
              }
            } else {
              result[0] += 0.056593159599191215;
            }
          } else {
            result[0] += 0.0626587462687593;
          }
        }
      } else {
        result[0] += -0.05742791911844697;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.85305833816528498) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.11189549779017988;
              } else {
                result[0] += -0.003726607784587021;
              }
            } else {
              result[0] += 0.033048469820725956;
            }
          } else {
            result[0] += 0.07849087494843293;
          }
        } else {
          result[0] += -0.052068253332175664;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.10916561688129658;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                result[0] += -0.0028712740949070827;
              } else {
                result[0] += -0.03896892566665362;
              }
            } else {
              result[0] += -0.0464401238333363;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
              result[0] += 0.029959557490747163;
            } else {
              result[0] += -0.027987720696210867;
            }
          } else {
            result[0] += -0.045370756408906095;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
      result[0] += 0.009647047433307764;
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)10.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.08914902246993724;
          } else {
            result[0] += -0.005101010651559913;
          }
        } else {
          result[0] += 0.00450289668275767;
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.003733545372222666;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
                      result[0] += 0.021891554259089216;
                    } else {
                      result[0] += 0.08184616480846436;
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.8154783248901385) ) ) {
                      result[0] += -0.048078861025613165;
                    } else {
                      result[0] += 0.02273808635627189;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0009641947088521931;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10940361022949396) ) ) {
                        result[0] += -0.01490979079966582;
                      } else {
                        result[0] += -0.05651017259692941;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += 0.01086323054428333;
                      } else {
                        result[0] += -0.017923837493762054;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.04565881727626899;
              } else {
                result[0] += 0.009002654674317106;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += -0.024900888070216926;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.01066170102955967;
              } else {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                  result[0] += -0.018950447491189293;
                } else {
                  result[0] += -0.0003596537193052518;
                }
              }
            }
          }
        } else {
          result[0] += 2.752497924301486e-05;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.013580216110387736;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.06711329411277848;
              } else {
                result[0] += -0.01956164268557037;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.825982809066773349) ) ) {
                  result[0] += -0.005505602794961605;
                } else {
                  result[0] += 0.02426604036835693;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.378218650817871982) ) ) {
                  result[0] += 0.04011797565890949;
                } else {
                  result[0] += 0.10123215817072438;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.86655306816101163) ) ) {
                  result[0] += 0.008801825907200233;
                } else {
                  result[0] += 0.06858958699737268;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                  result[0] += 0.0013488681866555604;
                } else {
                  result[0] += -0.039932671783562676;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.274755001068116123) ) ) {
              result[0] += 0.06026439117097304;
            } else {
              result[0] += -0.03787186132159437;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                    result[0] += -0.042825339443640774;
                  } else {
                    result[0] += 0.09300198918601335;
                  }
                } else {
                  result[0] += 0.1019720317789816;
                }
              } else {
                result[0] += 0.04925507852075234;
              }
            } else {
              result[0] += 0.009660096764057984;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
          result[0] += 0.027910464945893745;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.07649894537698593;
          } else {
            result[0] += 0.02295265989385645;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.011322972943953031;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0683592742498007;
            } else {
              result[0] += 0.022922720943177915;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.04187077275285797;
            } else {
              result[0] += 0.09259198142844183;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.044202481707187775;
            } else {
              result[0] += 0.3281960004361767;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.10070912659804632;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.006236618984366117;
                } else {
                  result[0] += 0.12694641021921213;
                }
              } else {
                result[0] += -0.03611979882291836;
              }
            } else {
              result[0] += -0.04307952145922389;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.06630153428628559;
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.006249847119778186;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.0029002855071311993;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                      result[0] += 0.0782161656146818;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.043341875076294833) ) ) {
                        result[0] += 0.16530227909246528;
                      } else {
                        result[0] += 0.006211178565684113;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += -0.03523860716211363;
              } else {
                result[0] += 0.09919611706668778;
              }
            }
          } else {
            result[0] += -0.04092723379465363;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += -0.0004268940322075826;
      } else {
        result[0] += -0.023606968036921798;
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.689592361450196201) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                  result[0] += 0.03062481646671308;
                } else {
                  result[0] += 0.09255392025113307;
                }
              } else {
                result[0] += 0.013564185163358281;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.0038548509070272653;
                  } else {
                    result[0] += -0.03427988217088017;
                  }
                } else {
                  result[0] += 0.056565724408278076;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.109050035476685458) ) ) {
                    result[0] += -0.00790566200636572;
                  } else {
                    result[0] += 0.0275657860557331;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                    result[0] += 0.00541864996879778;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                      result[0] += 0.03380840076354148;
                    } else {
                      result[0] += 0.016283147529889495;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.007109356513981358;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.032892662006645625;
            } else {
              result[0] += 0.00281209482242805;
            }
          } else {
            result[0] += -0.027461428241963567;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
            result[0] += -0.033201373579475245;
          } else {
            result[0] += 0.0076236150945127815;
          }
        } else {
          result[0] += -0.04849565101513463;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.02188651954416713;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                result[0] += -0.03693531016500045;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.07030505392258875;
                } else {
                  result[0] += -0.009926358644584424;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.802100181579590732) ) ) {
                  result[0] += -0.003585781553820735;
                } else {
                  result[0] += 0.026588772668145935;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.06968032433025408;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                    result[0] += 0.023518699259064517;
                  } else {
                    result[0] += 0.08976120210305934;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.021267555233364828;
                } else {
                  result[0] += 0.015022929659541835;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.549068689346314365) ) ) {
                    result[0] += 0.011648067853774566;
                  } else {
                    result[0] += 0.05094989455765151;
                  }
                } else {
                  result[0] += 0.11108150781423826;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.059252197833170844;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.0073153237687281145;
                } else {
                  result[0] += 0.03318388495505371;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += -0.060663383100417384;
                } else {
                  result[0] += 0.046881463598315014;
                }
              }
            } else {
              result[0] += 0.12843149374294452;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
            result[0] += -0.03153408425187495;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.033017667825111695;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.827628135681153232) ) ) {
                  result[0] += 0.005449253864953543;
                } else {
                  result[0] += -0.10851459155952492;
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.009790710226932708;
              } else {
                result[0] += 0.07534636906866758;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.07206906984231254;
            } else {
              result[0] += 0.025932256162788565;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.561026811599732333) ) ) {
                  result[0] += 0.04100168112059963;
                } else {
                  result[0] += -0.033713390551789506;
                }
              } else {
                result[0] += -0.026296457153426135;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.448852539062500444) ) ) {
                result[0] += 0.08300572212387661;
              } else {
                result[0] += -0.03328785686363678;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.85305833816528498) ) ) {
          result[0] += 0.03414566976190469;
        } else {
          result[0] += -0.03312988355041009;
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.0976056792888794;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                result[0] += -0.0036991771364006903;
              } else {
                result[0] += -0.056018626167053436;
              }
            } else {
              result[0] += -0.04043986682726088;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.03587831724849737;
              } else {
                result[0] += 0.14238053847619064;
              }
            } else {
              result[0] += -0.036737017640542935;
            }
          } else {
            result[0] += -0.02098038086906928;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.000482738160809223;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.249904870986938921) ) ) {
          result[0] += -0.024414360669851153;
        } else {
          result[0] += 0.06508952945492205;
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                  result[0] += 0.028130179871499142;
                } else {
                  result[0] += 0.0844193935633512;
                }
              } else {
                result[0] += 0.013109630305596484;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
                      result[0] += 0.0229755631592142;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.07286426023806163;
                        } else {
                          result[0] += -0.007502186853938591;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                          result[0] += 0.0072355770888543934;
                        } else {
                          result[0] += -0.03448377214144918;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                      result[0] += -0.0004122038687634493;
                    } else {
                      result[0] += -0.054823142977453124;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.901921629905701128) ) ) {
                    result[0] += 0.057164403694736354;
                  } else {
                    result[0] += -0.0458415388364493;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.551017761230469638) ) ) {
                  result[0] += -0.009426236339739373;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.034434640863116206;
                    } else {
                      result[0] += 5.751273293930357e-05;
                    }
                  } else {
                    result[0] += 0.018736049751926356;
                  }
                }
              }
            }
          } else {
            result[0] += -0.006805287125626511;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
              result[0] += -0.043928209810303376;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.334978580474854404) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.003706111505281088;
                } else {
                  result[0] += 0.01696815452075514;
                }
              } else {
                result[0] += 0.020813624659129344;
              }
            }
          } else {
            result[0] += -0.02525984051289964;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          result[0] += -0.005493959098680645;
        } else {
          result[0] += -0.0453154234580901;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
              result[0] += -0.016395870895408338;
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += 0.019566816623074988;
              } else {
                result[0] += 0.0009571021716654773;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.000307083129883701) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    result[0] += -0.023854114168432694;
                  } else {
                    result[0] += 0.04476902041687936;
                  }
                } else {
                  result[0] += 0.052397564321814874;
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.042520842738711845;
                } else {
                  result[0] += 0.08642291015855914;
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += -0.027207482811632485;
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.09874723567900491;
                  } else {
                    result[0] += -0.044657769313515595;
                  }
                }
              } else {
                result[0] += 0.024553162012221914;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.04812066001943191;
            } else {
              result[0] += 0.027474164719938205;
            }
          } else {
            result[0] += 0.008903587968915804;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
            result[0] += -0.032548969013855844;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.03026438078844358;
              } else {
                result[0] += -0.000880848612364058;
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.01682788776245891;
              } else {
                result[0] += 0.0764046737710854;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.06803609510900273;
            } else {
              result[0] += -0.0019961888539666273;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += 0.05900938335505407;
            } else {
              result[0] += -0.04425046889538477;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.65464162826538264) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += 0.005013762083064614;
            } else {
              result[0] += 0.1618567625476451;
            }
          } else {
            result[0] += -0.05494118573666545;
          }
        } else {
          result[0] += 0.037966086249890375;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.08902608930423961;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                result[0] += -0.0030256150339063527;
              } else {
                result[0] += -0.05199685061275549;
              }
            } else {
              result[0] += -0.03675535045409435;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.36324071884155451) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.03585078063681521;
              } else {
                result[0] += 0.14924666421415092;
              }
            } else {
              result[0] += -0.030608150469413172;
            }
          } else {
            result[0] += -0.018966606592693402;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.0005349714676791145;
      } else {
        result[0] += -0.020683593974889836;
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
              result[0] += 0.029713784842648305;
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.0026826483661781956;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
                      result[0] += 0.024191740346814827;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                          result[0] += 0.028200350178561498;
                        } else {
                          result[0] += -0.04116051243608191;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
                          result[0] += -0.06306592325922404;
                        } else {
                          result[0] += 0.0327472040979489;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.058245732220717696;
                  }
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0012172091774754572;
                  } else {
                    result[0] += 0.03262307191823046;
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.008996543730790162;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                        result[0] += -0.09507998072505136;
                      } else {
                        result[0] += -0.03049203720650695;
                      }
                    }
                  } else {
                    result[0] += 0.02628157107005239;
                  }
                }
              }
            }
          } else {
            result[0] += -0.006320076896847455;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
              result[0] += -0.040952843160636306;
            } else {
              result[0] += 0.0017828066397346087;
            }
          } else {
            result[0] += -0.023095743908743364;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
            result[0] += 0.01165404489825644;
          } else {
            result[0] += -0.02922089727155045;
          }
        } else {
          result[0] += -0.04548116110402232;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.917405366897583452) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.568724632263184482) ) ) {
              result[0] += -0.004399085544938892;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.549068689346314365) ) ) {
                  result[0] += 0.014690688557080129;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
                    result[0] += 0.06090149828273424;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                      result[0] += 0.03251853109278342;
                    } else {
                      result[0] += -0.03208545112978719;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
                    result[0] += 0.0013712320411148687;
                  } else {
                    result[0] += -0.06238985310912748;
                  }
                } else {
                  result[0] += 0.006567129471282282;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      result[0] += -0.023024411077299245;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += 0.017024847173641954;
                      } else {
                        result[0] += 0.09255390275787177;
                      }
                    }
                  } else {
                    result[0] += 0.07269825778677272;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                    result[0] += 0.032908561262436936;
                  } else {
                    result[0] += -0.044719046105329896;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.138696432113648349) ) ) {
                  result[0] += 0.025417878635903553;
                } else {
                  result[0] += 0.07698673392688501;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.804059982299805576) ) ) {
                      result[0] += -0.021912082306623178;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                        result[0] += 0.09300448159299192;
                      } else {
                        result[0] += -0.004801716890913892;
                      }
                    }
                  } else {
                    result[0] += 0.09407430903524065;
                  }
                } else {
                  result[0] += -0.04683543818036834;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
                  result[0] += -0.021126173645740716;
                } else {
                  result[0] += 0.031249313370313527;
                }
              }
            }
          }
        } else {
          result[0] += -0.04894400538982279;
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.447260618209839755) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.0063688163752330315;
                } else {
                  result[0] += 0.07630721174092285;
                }
              } else {
                result[0] += 0.053970888248820736;
              }
            } else {
              result[0] += 0.0054500811625254626;
            }
          } else {
            result[0] += 0.0788512852373078;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.274755001068116123) ) ) {
            result[0] += 0.054725663439610384;
          } else {
            result[0] += -0.03237134826889614;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
        result[0] += 0.03668225688900211;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.09297395928741767;
              } else {
                result[0] += 0.0059796379393975245;
              }
            } else {
              result[0] += -0.030553628784830457;
            }
          } else {
            result[0] += -0.05058702731100911;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.138696432113648349) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.36324071884155451) ) ) {
              result[0] += 0.0685600610542048;
            } else {
              result[0] += -0.013365445915595972;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.014952359673986775;
            } else {
              result[0] += 0.25956026314929237;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY(  (data[31].missing != -1) && (data[31].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.01503259430924208;
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                      result[0] += 0.008225889247979365;
                    } else {
                      result[0] += -0.004878566438378058;
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.985194206237793857) ) ) {
                      result[0] += -0.0033602489502644207;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.004022245915976482;
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                            result[0] += 0.09672790446487638;
                          } else {
                            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                                result[0] += 0.049529016114026894;
                              } else {
                                result[0] += -0.07439364529458231;
                              }
                            } else {
                              result[0] += -0.03341163828451138;
                            }
                          }
                        } else {
                          result[0] += 0.06356893975732125;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.02005251756704255;
                      } else {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.004586791424993826;
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                            result[0] += 0.06243528578731871;
                          } else {
                            result[0] += 0.27148681072363867;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.028055120145628516;
                    }
                  } else {
                    result[0] += -0.06080697584457804;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
                      result[0] += -0.04351896453630356;
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.06893053025372696;
                      } else {
                        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                          result[0] += 0.010860707787098077;
                        } else {
                          result[0] += -0.06570981019958569;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.004743263033788156;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.238486170768738237) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.0027039973611558557;
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                            result[0] += -0.035735252308105425;
                          } else {
                            result[0] += -0.07832091773935268;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)9.167253971099855292) ) ) {
                          result[0] += -0.010965666400715016;
                        } else {
                          result[0] += -0.19809921246940987;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.198464870452881303) ) ) {
                          result[0] += 0.022722405822213715;
                        } else {
                          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                            result[0] += -0.0019695731784554163;
                          } else {
                            result[0] += -0.04887729173799346;
                          }
                        }
                      } else {
                        result[0] += 0.04122152794531916;
                      }
                    }
                  } else {
                    result[0] += 0.034638096883451826;
                  }
                }
              }
            } else {
              result[0] += -0.000371598855206712;
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += -0.021392142678027692;
          } else {
            result[0] += -0.10281924813583102;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
          result[0] += -0.029835088933935994;
        } else {
          result[0] += 0.004064163137573506;
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
            result[0] += 0.008912360362912408;
          } else {
            result[0] += -0.005643426367261686;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += -0.0548524753873501;
              } else {
                result[0] += -0.005846149012105405;
              }
            } else {
              result[0] += 0.02710587341995731;
            }
          } else {
            result[0] += 0.012279344610194816;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
            result[0] += -0.03588236988399545;
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += -0.037201673419938484;
            } else {
              result[0] += 0.021317764476584938;
            }
          }
        } else {
          result[0] += -0.039148232471935836;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
            result[0] += -0.014620257516379255;
          } else {
            result[0] += 0.008267598257616655;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.318498134613038886) ) ) {
              result[0] += 0.038471274282108364;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += -0.007021821659056952;
              } else {
                result[0] += 0.21623626412274619;
              }
            }
          } else {
            result[0] += 0.007776694151278557;
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
          result[0] += 0.032267068675156496;
        } else {
          result[0] += -0.024595863250175397;
        }
      }
    } else {
      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.035591147731184145;
        } else {
          result[0] += -0.0011128739495805437;
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.013302994302237275;
            } else {
              result[0] += -0.04546158784186522;
            }
          } else {
            result[0] += -0.06923228899854321;
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.681325197219849521) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.00996560749633446;
              } else {
                result[0] += 0.06020182453212878;
              }
            } else {
              result[0] += -0.01892431212509345;
            }
          } else {
            result[0] += -0.01746302422171624;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.0005594647635171567;
        } else {
          result[0] += -0.03642212691284877;
        }
      } else {
        result[0] += -0.021230699189830886;
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += 0.04334013804505234;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                      result[0] += 0.013536694093199013;
                    } else {
                      result[0] += -0.043722476946779255;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += 0.019272144297586297;
                  } else {
                    result[0] += 0.04836576245314662;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.21245336532592951) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.583132982254028764) ) ) {
                      result[0] += 9.735171166590528e-05;
                    } else {
                      result[0] += 0.06870325338328388;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                      result[0] += 0.11482361315010735;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.059239812080181656;
                      } else {
                        result[0] += -0.005058738644773476;
                      }
                    }
                  }
                } else {
                  result[0] += -0.00852560071124926;
                }
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.010768422233017301;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.092434883117676669) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                        result[0] += -0.02884926174126038;
                      } else {
                        result[0] += 0.04606094937249461;
                      }
                    } else {
                      result[0] += -0.008126265999853298;
                    }
                  } else {
                    result[0] += 0.0578687670619718;
                  }
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.01625398337986932;
                    } else {
                      result[0] += 0.035645993598320234;
                    }
                  } else {
                    result[0] += -0.04196867596482096;
                  }
                } else {
                  result[0] += 0.02078458731635349;
                }
              }
            }
          } else {
            result[0] += -0.0053212093252170585;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.847910165786744052) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.33441734313965021) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.006788702568679458;
                      } else {
                        result[0] += 0.02206906922105978;
                      }
                    } else {
                      result[0] += -0.04643994078235187;
                    }
                  } else {
                    result[0] += -0.019382668725691464;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.04507210154788481;
                  } else {
                    result[0] += -0.022313115472290736;
                  }
                }
              } else {
                result[0] += 0.013388721711574734;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.024566603778567772;
              } else {
                result[0] += -0.008951347904576128;
              }
            }
          } else {
            result[0] += -0.021408860508524884;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.363078355789185458) ) ) {
            result[0] += -0.03479176811917455;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.022065387788453742;
            } else {
              result[0] += -0.020390263995465348;
            }
          }
        } else {
          result[0] += -0.03967865207392979;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
            result[0] += 5.019958242857362e-05;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.02165043080974941;
              } else {
                result[0] += 0.05262463748914349;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                result[0] += 0.00839613607166252;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.0765311233573563;
                } else {
                  result[0] += -0.010544489295299856;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.08141032776692247;
                  } else {
                    result[0] += 0.009587071078023772;
                  }
                } else {
                  result[0] += 0.12052589084995441;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                  result[0] += 0.03452266469303634;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.007923076019457152;
                    } else {
                      result[0] += 0.08265490424526158;
                    }
                  } else {
                    result[0] += -0.0423594002297206;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.05366659828009934;
              } else {
                result[0] += 0.1534702204128121;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.016970814130417167;
              } else {
                result[0] += -0.04126910926275607;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
                result[0] += -0.005976541933328721;
              } else {
                result[0] += 0.03224147876748942;
              }
            }
          }
        }
      } else {
        result[0] += -0.03732004395250726;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
          result[0] += -0.03289391168045965;
        } else {
          result[0] += 0.03693218955966169;
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.027914600589544314;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
            result[0] += 0.030520549781775538;
          } else {
            result[0] += -0.014793111781721267;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY(  (data[49].missing != -1) && (data[49].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.010246967503262205;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                      result[0] += 0.019290057931310205;
                    } else {
                      result[0] += -0.04402078673480567;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.942183732986451083) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
                      result[0] += -0.012056046723850541;
                    } else {
                      result[0] += -0.03142165885061015;
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                          result[0] += -0.030266367971114323;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                            result[0] += -0.034519449064859216;
                          } else {
                            result[0] += 0.044304199927245005;
                          }
                        }
                      } else {
                        result[0] += -0.001357652681805427;
                      }
                    } else {
                      result[0] += 0.04269429883996042;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                    result[0] += 0.010467378234377482;
                  } else {
                    result[0] += -0.008489313819165336;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
                    result[0] += -0.011250793476166391;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                        result[0] += 0.013528974709605669;
                      } else {
                        result[0] += 0.10829918232686822;
                      }
                    } else {
                      result[0] += 0.013145445004587239;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.018307300274656803;
            }
          } else {
            result[0] += -0.00036270278108033253;
          }
        } else {
          result[0] += -0.032448181864903024;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.586156606674195224) ) ) {
          result[0] += -0.012795139814060781;
        } else {
          result[0] += -0.05374512288858228;
        }
      }
    } else {
      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.112327098846436435) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
            result[0] += 0.05158664031493296;
          } else {
            result[0] += -0.011744008469431053;
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.013364706399164381;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.589327573776246005) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.026624068805922198;
                  } else {
                    result[0] += -0.019913998578441238;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.960975408554078037) ) ) {
                    result[0] += -0.0159220667630136;
                  } else {
                    result[0] += -0.0659509176521242;
                  }
                }
              } else {
                result[0] += 0.0012659433854707177;
              }
            } else {
              result[0] += 0.022602724989336506;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.002974671260648413;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
                result[0] += 0.011458489916120435;
              } else {
                result[0] += 0.036971623386492666;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += 0.01687828400005009;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                result[0] += 0.10066891628311057;
              } else {
                result[0] += 0.032360922078658;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.0006604349700179208;
          } else {
            result[0] += -0.04317159650705286;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
            result[0] += 0.002523517986646485;
          } else {
            result[0] += -0.038217988025009364;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.917405366897583452) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.03741453399152664;
            } else {
              result[0] += 0.013345416428934598;
            }
          } else {
            result[0] += -0.025179742905362974;
          }
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                result[0] += 0.02643866037213386;
              } else {
                result[0] += 0.05713795170884586;
              }
            } else {
              result[0] += 0.015750131261492946;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
              result[0] += 0.03413616126646043;
            } else {
              result[0] += -0.0035754324546478167;
            }
          }
        } else {
          result[0] += 0.04582166756388084;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.9054608345031756) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.0659961582891898;
            } else {
              result[0] += 0.008102995026399043;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.018936952452942794;
              } else {
                result[0] += 0.05389666636228783;
              }
            } else {
              result[0] += 0.01167291676365622;
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.04135957523240552;
          } else {
            result[0] += 0.3118247491282904;
          }
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.1094134956352175;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06685587812770812;
              } else {
                result[0] += -0.025537793377253343;
              }
            } else {
              result[0] += -0.015006265355139492;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
              result[0] += 0.05675837647788358;
            } else {
              result[0] += 0.00024376901306859134;
            }
          } else {
            result[0] += -0.007546618501562976;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.0005602118597923936;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                result[0] += -0.10156151041580658;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                  result[0] += 0.05271974108173977;
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.08745349574436498;
                  } else {
                    result[0] += 0.08783424822163048;
                  }
                }
              }
            } else {
              result[0] += 0.09838573590124722;
            }
          } else {
            result[0] += -0.02342199654641325;
          }
        } else {
          result[0] += -0.03077667876762685;
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.03621426659068632;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.01518293034791777;
                } else {
                  result[0] += -0.010124537255325389;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
                    result[0] += 0.006580782382189971;
                  } else {
                    result[0] += -0.03323001229789517;
                  }
                } else {
                  result[0] += 0.09716567212112542;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.223051309585572177) ) ) {
                  result[0] += 0.04509410979908743;
                } else {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += 0.04518902874739721;
                      } else {
                        result[0] += 0.009368524616573885;
                      }
                    } else {
                      result[0] += 0.005817391961030479;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += -0.015708540174443033;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.03299538771080913;
                      } else {
                        result[0] += -0.019902470008223178;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.870983839035034624) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  result[0] += 0.01706660120665787;
                } else {
                  result[0] += -0.03985805235545414;
                }
              } else {
                result[0] += 0.0016173713925889078;
              }
            } else {
              result[0] += 0.032900196708203884;
            }
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
                    result[0] += -0.0031961180046461094;
                  } else {
                    result[0] += -0.01872679702393358;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                        result[0] += -0.034980702551774163;
                      } else {
                        result[0] += 0.05682908934020001;
                      }
                    } else {
                      result[0] += -0.07364858380800503;
                    }
                  } else {
                    result[0] += -0.019689481265184206;
                  }
                }
              } else {
                result[0] += 0.013558038955395547;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.020357125269660784;
              } else {
                result[0] += -0.007670186962989461;
              }
            }
          } else {
            result[0] += -0.0194302108130013;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
            result[0] += -0.02474947464992258;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
              result[0] += -0.011408629095379298;
            } else {
              result[0] += 0.03321665417403629;
            }
          }
        } else {
          result[0] += -0.03407083320959691;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.966960191726685458) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.01202948375117848;
                } else {
                  result[0] += 0.01933240949978896;
                }
              } else {
                result[0] += -0.0017926771836539006;
              }
            } else {
              result[0] += -0.011523025273864836;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
              result[0] += 0.02830143661448672;
            } else {
              result[0] += -0.014502251943346204;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.014199625136393668;
              } else {
                result[0] += 0.04382260827241066;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                result[0] += 0.040054955620585926;
              } else {
                result[0] += 0.13832765766022462;
              }
            }
          } else {
            result[0] += 0.01280638204638824;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
          result[0] += 0.021872231252114016;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.932935476303101474) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.10575018787078148;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                  result[0] += 0.0018836510103614997;
                } else {
                  result[0] += -0.03473394722962549;
                }
              }
            } else {
              result[0] += -0.060458126128179385;
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.004154391983988487;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.012639545298979295;
              } else {
                result[0] += 0.062031760160672925;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
        result[0] += 0.03600950865365942;
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.023002135575679646;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.768316030502320224) ) ) {
              result[0] += 0.014346251699744404;
            } else {
              result[0] += -0.029636676168573933;
            }
          }
        } else {
          result[0] += 0.15492740760643012;
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
              result[0] += -0.007824506472567279;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.705446481704712802) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += -0.0006483764716950234;
                  } else {
                    result[0] += -0.07874737780851054;
                  }
                } else {
                  result[0] += 0.03120644194578473;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.01241322019213223;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.680079460144043857) ) ) {
                    result[0] += 0.026478160009326585;
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.09364319642575535;
                    } else {
                      result[0] += 0.026627319382367516;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                  result[0] += 0.006733931705195203;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.009414861205029115;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                        result[0] += 0.021970383155394336;
                      } else {
                        result[0] += -0.0046417388233368885;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                          result[0] += -0.013365044889718633;
                        } else {
                          result[0] += -0.039208728321726796;
                        }
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.740319490432739702) ) ) {
                          result[0] += -0.05120354329428933;
                        } else {
                          result[0] += -0.0013192123174845596;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.019278348404437044;
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.00743917742586942;
                        } else {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.740319490432739702) ) ) {
                            result[0] += 0.04113837862413611;
                          } else {
                            result[0] += 0.13631140599971983;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54296922683715998) ) ) {
                      result[0] += 0.005670906715691846;
                    } else {
                      result[0] += -0.064413476199121;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                        result[0] += -0.03570800766255482;
                      } else {
                        result[0] += 0.01631900467358242;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += 0.0664375619275182;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.0017262405485359619;
                        } else {
                          result[0] += 0.03493370380132732;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.297559976577759233) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.06047577506764884;
                    } else {
                      result[0] += -0.056942535911502;
                    }
                  } else {
                    result[0] += -0.054426070916204966;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.46093606948852717) ) ) {
                      result[0] += -0.03003650395386387;
                    } else {
                      result[0] += 0.02373529061177639;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.740319490432739702) ) ) {
                        result[0] += 0.1544861287694355;
                      } else {
                        result[0] += -0.08399705457753942;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                        result[0] += -0.03639193279602027;
                      } else {
                        result[0] += -0.07997427351611777;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
                    result[0] += -0.016609975117903165;
                  } else {
                    result[0] += 0.04720756825173537;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.745876312255860263) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                      result[0] += 0.10140006769043029;
                    } else {
                      result[0] += -0.188253627730048;
                    }
                  } else {
                    result[0] += -0.14322031964031956;
                  }
                } else {
                  result[0] += 0.038645220198965755;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.01027917100818829;
            } else {
              result[0] += -0.026055037589213945;
            }
          } else {
            result[0] += 0.02726285349523134;
          }
        }
      } else {
        result[0] += -0.0003630657349937397;
      }
    } else {
      result[0] += 0.002419545708535488;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += 0.0016353091174708357;
        } else {
          result[0] += 0.019185375930020485;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
          result[0] += 0.03194838782282888;
        } else {
          result[0] += 0.014721515284334531;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
              result[0] += 0.004781772246101301;
            } else {
              result[0] += -0.040458194158363596;
            }
          } else {
            result[0] += 0.02349102958605904;
          }
        } else {
          result[0] += 0.0715136580952541;
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.08022330758858132;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.009872053833415478;
                } else {
                  result[0] += 0.10634043965793301;
                }
              } else {
                result[0] += -0.023529799902591894;
              }
            } else {
              result[0] += -0.02585791046472166;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += 0.03725646368482274;
              } else {
                result[0] += -0.04739588407383853;
              }
            } else {
              result[0] += -0.0011934926868105136;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.869292974472046787) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              result[0] += 0.0022941567735143185;
            } else {
              result[0] += -0.02370605916867279;
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += 0.007191474023873068;
                } else {
                  result[0] += -0.04536376311053295;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.673553824424744096) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.01025499384574021;
                    } else {
                      result[0] += -0.03851374577861946;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.034017702370943856;
                    } else {
                      result[0] += 0.015858874965328652;
                    }
                  }
                } else {
                  result[0] += 0.046678000927023615;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.4822273254394549) ) ) {
                result[0] += -0.019863007650693945;
              } else {
                result[0] += 0.015947875505679693;
              }
            }
          }
        } else {
          result[0] += 0.0008975824886619985;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.007760857390852877;
            } else {
              result[0] += -0.010836028795248809;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.007482851602658003;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                  result[0] += -0.04381596678240233;
                } else {
                  result[0] += -0.016848717623407002;
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.05516915170414935;
                  } else {
                    result[0] += 0.013568540835315033;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
                    result[0] += -0.08111220705907399;
                  } else {
                    result[0] += 0.009043230161140762;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05539035797119318) ) ) {
                    result[0] += -0.01735925671465103;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.07178725322627677;
                    } else {
                      result[0] += -0.018522619447248766;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                    result[0] += -0.01103813446088403;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.015285742006100254;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.05155742806600555;
                        } else {
                          result[0] += 0.053600699847830224;
                        }
                      } else {
                        result[0] += 0.006250204456366978;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.05025571056212359;
            } else {
              result[0] += -0.0004017781531838625;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
              result[0] += 0.029096939048428607;
            } else {
              result[0] += 0.07814754979818732;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
        result[0] += -0.006238610875439345;
      } else {
        result[0] += -0.00015727075755252545;
      }
    }
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.400584220886231357) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
            result[0] += -0.007744828403970713;
          } else {
            result[0] += -0.04485470364753025;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.03534758159239341;
          } else {
            result[0] += -0.09978301330017846;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
                result[0] += -0.00660499537705635;
              } else {
                result[0] += 0.015489551839351216;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                result[0] += 0.04967541063517912;
              } else {
                result[0] += -0.010238356861271612;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
                  result[0] += 0.004902374221048764;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                    result[0] += 0.05704678116160006;
                  } else {
                    result[0] += 0.010379820214504643;
                  }
                }
              } else {
                result[0] += 0.10154950906670085;
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.011676114253059394;
              } else {
                result[0] += 0.06250080682781366;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
            result[0] += 0.014627970850672087;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.09008685202448047;
            } else {
              result[0] += -0.0024333693392711835;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.373361587524414951) ) ) {
          result[0] += 0.01437547523858708;
        } else {
          result[0] += 0.06176567892686287;
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.07290077826786916;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.329718828201294833) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.86655306816101163) ) ) {
                  result[0] += -0.0024377264773390605;
                } else {
                  result[0] += 0.06982594234437786;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.650573849678039995) ) ) {
                  result[0] += -0.02627445131336369;
                } else {
                  result[0] += 0.17557890377485794;
                }
              }
            } else {
              result[0] += -0.04780492672631029;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.673553824424744096) ) ) {
                result[0] += 0.04271803376872095;
              } else {
                result[0] += -0.018662852238671614;
              }
            } else {
              result[0] += -0.015108451201032064;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.0067774476342178595;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.010169514169437923;
              } else {
                result[0] += -0.009753652015300375;
              }
            } else {
              result[0] += 0.02314425993004545;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.03558944251462882;
                } else {
                  result[0] += -0.008012870387099936;
                }
              } else {
                result[0] += 0.03507136914347136;
              }
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.010564153805641033;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.032515442987272276;
                } else {
                  result[0] += -0.0010104835133223165;
                }
              }
            }
          } else {
            result[0] += -0.0009653562858294991;
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.007007219515463201;
            } else {
              result[0] += -0.010150351144260618;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.601370334625245029) ) ) {
                  result[0] += -0.0077776154437960154;
                } else {
                  result[0] += -0.034193300868635444;
                }
              } else {
                result[0] += -0.024915626383348148;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.028310101402878148;
                    } else {
                      result[0] += 0.0560346645945054;
                    }
                  } else {
                    result[0] += 0.011562896545914974;
                  }
                } else {
                  result[0] += -0.02772594525916033;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.025295187805644038;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.010454387523829371;
                  } else {
                    result[0] += 0.013198799496715138;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.913499355316162998) ) ) {
            result[0] += -0.014134705693318523;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.314458370208742011) ) ) {
                result[0] += -0.04649976995157376;
              } else {
                result[0] += 0.03636349037210666;
              }
            } else {
              result[0] += 0.04119695529165537;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                result[0] += 0.0013331494161819264;
              } else {
                result[0] += -0.025554381258435194;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.00039782005039025627;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.02755926551080931;
                    } else {
                      result[0] += 0.0012260154010384247;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.04777017715336395;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += -0.00047176450244350884;
                    } else {
                      result[0] += 0.07388258603660706;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.08364865117048886;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                    result[0] += -0.0229844645923621;
                  } else {
                    result[0] += 0.019797763261749857;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.0061179665196967985;
                } else {
                  result[0] += -0.1072681336948619;
                }
              } else {
                result[0] += -0.08023770210567473;
              }
            } else {
              result[0] += -0.00428995818740147;
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.11301071292981649;
            } else {
              result[0] += -0.001599084653871225;
            }
          } else {
            result[0] += -0.056030916277823174;
          }
        }
      } else {
        result[0] += 0.00044547696926893566;
      }
    }
  } else {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.109050035476685458) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.007032190858286673;
            } else {
              result[0] += -0.09467563534755366;
            }
          } else {
            result[0] += 0.009488456476141315;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
            result[0] += 0.04085730768450129;
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                  result[0] += -0.0019652447894021765;
                } else {
                  result[0] += -0.06569191060542684;
                }
              } else {
                result[0] += 0.010838403374851757;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
                result[0] += 0.010701805937812087;
              } else {
                result[0] += 0.03512569630001748;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.053021186028184925;
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.011959383179219137;
          } else {
            result[0] += 0.004305075840028902;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.166635274887085849) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.663379430770874912) ) ) {
            result[0] += 0.02503375074881515;
          } else {
            result[0] += 0.06202480703841891;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
            result[0] += 0.022074593602021866;
          } else {
            result[0] += -0.011439297552540025;
          }
        }
      } else {
        result[0] += -0.010244908427418901;
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.011960850067363903;
      } else {
        result[0] += 0.11648533835100168;
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.007459537673165777;
                } else {
                  result[0] += -0.06221399730806118;
                }
              } else {
                result[0] += -0.006108599643814757;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                result[0] += -0.044975281012544784;
              } else {
                result[0] += -0.008947192982021917;
              }
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
                  result[0] += 0.015905045221816863;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.03260142964479184;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.014098943870877595;
                    } else {
                      result[0] += 0.05978517077468202;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                  result[0] += 0.013634986024092223;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                    result[0] += -0.03271354090841687;
                  } else {
                    result[0] += 0.04432888941264453;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                result[0] += -0.014390466588685785;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.02644761027546721;
                  } else {
                    result[0] += -0.041735761790712485;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                    result[0] += -0.005334580256199607;
                  } else {
                    result[0] += 0.01843223425136185;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
            result[0] += -0.00013688692821369145;
          } else {
            result[0] += -0.02920940993370069;
          }
        }
      } else {
        result[0] += 0.002059577697259547;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                  result[0] += -0.0023425966696655727;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.005615616835429385;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.008042573077176261;
                      } else {
                        result[0] += 0.02451743520607947;
                      }
                    }
                  } else {
                    result[0] += -0.0067273751563893805;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                    result[0] += 0.013134187139706103;
                  } else {
                    result[0] += -0.053782409727365976;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.917705297470093662) ) ) {
                    result[0] += -0.006375235881793806;
                  } else {
                    result[0] += -0.030490038461344234;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += -0.002759512570490047;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.05622703251004059;
                      } else {
                        result[0] += -0.0054580967811592475;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.99033999443054288) ) ) {
                        result[0] += 0.021455432747721135;
                      } else {
                        result[0] += 0.07016494477627012;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.003838300704956943) ) ) {
                        result[0] += 0.002966077750965624;
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                          result[0] += 0.0828349213177888;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.03291656313591308;
                          } else {
                            result[0] += 0.08233436337680551;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.05412923208867957;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.036931514739991123) ) ) {
                      result[0] += 0.059123495061900004;
                    } else {
                      result[0] += -0.014998730023139764;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.382196187973023349) ) ) {
                      result[0] += 0.002503493964767272;
                    } else {
                      result[0] += 0.03601819921462272;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
              result[0] += 0.023163327351508448;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                result[0] += 0.0033703220261384853;
              } else {
                result[0] += -0.03643184346087846;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.01686408828203533;
            } else {
              result[0] += 0.10928032952614697;
            }
          } else {
            result[0] += 0.08615196232347946;
          }
        }
      } else {
        result[0] += -0.019325919591119678;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.768316030502320224) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += 0.003693360262354728;
          } else {
            result[0] += 0.09813272208726001;
          }
        } else {
          result[0] += 0.08256416771517824;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0742707060337032;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.021008318739639254;
                } else {
                  result[0] += 0.13610530902861792;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += 0.009286617283552323;
              } else {
                result[0] += -0.023522708960004683;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += 0.05354093599532517;
            } else {
              result[0] += -0.008810291154556315;
            }
          }
        } else {
          result[0] += 0.13565996527224153;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40695333480835139) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
            result[0] += 0.008961841843405626;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
              result[0] += 0.008524804599039279;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.004885030541994476;
              } else {
                result[0] += -0.03792643710287045;
              }
            }
          }
        } else {
          result[0] += 0.024356050873321593;
        }
      } else {
        result[0] += -0.01596194579827158;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.373361587524414951) ) ) {
          result[0] += 0.003933961568834052;
        } else {
          result[0] += -0.017092746686101907;
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0321340233166375;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.556798219680787021) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += -0.1036698448649323;
              } else {
                result[0] += -0.019839716553554296;
              }
            } else {
              result[0] += -0.05610421485726175;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.353313446044923651) ) ) {
            result[0] += 0.012159115893859327;
          } else {
            result[0] += -0.016459753822443978;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.0009538952310734445;
      } else {
        if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += -0.0011041308134319009;
              } else {
                result[0] += -0.040113115379925054;
              }
            } else {
              result[0] += -0.04159093083881199;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                    result[0] += -0.0015370492545759872;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.015356652011212292;
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.07704133682284026;
                      } else {
                        result[0] += -0.012564380900126185;
                      }
                    }
                  }
                } else {
                  result[0] += -0.05573422417729479;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.99033999443054288) ) ) {
                    result[0] += 0.011298680568997877;
                  } else {
                    result[0] += 0.07627073548616192;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.040083281130616655;
                  } else {
                    result[0] += 0.01765523258576827;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0862073779263987;
                      } else {
                        result[0] += -0.0688034575005103;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.42895507812500178) ) ) {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.009948663907756478;
                        } else {
                          result[0] += -0.03463145752756696;
                        }
                      } else {
                        result[0] += 0.02136865441981353;
                      }
                    }
                  } else {
                    result[0] += -0.018664597496931706;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += 0.046439955888563866;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.917405366897583452) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.005923931395603835;
                      } else {
                        result[0] += 0.04587591619398455;
                      }
                    } else {
                      result[0] += -0.05896827304917682;
                    }
                  }
                }
              } else {
                result[0] += -0.022177181607132834;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
            result[0] += -0.05641143492908282;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += -0.045194489726094834;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                      result[0] += -0.017769885968894273;
                    } else {
                      result[0] += 0.013582451434173255;
                    }
                  } else {
                    result[0] += -0.03900744254111263;
                  }
                }
              } else {
                result[0] += -0.04782108611445072;
              }
            } else {
              result[0] += -0.0032406211791789043;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += -0.001508109312227451;
        } else {
          result[0] += -0.03466320586119126;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.0038996358085048653;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.722943305969239169) ) ) {
                  result[0] += -0.00897625677301137;
                } else {
                  result[0] += 0.013931377494489847;
                }
              }
            } else {
              result[0] += -0.029093545418687805;
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.04591097323335009;
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.019931779077387118;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.015411587296425706;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.07424366061561842;
                    } else {
                      result[0] += -0.024603499294508984;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.013620393729021832;
                      } else {
                        result[0] += -0.10650925773980223;
                      }
                    } else {
                      result[0] += -0.026625749119141225;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.847910165786744052) ) ) {
                result[0] += 0.003326753807240044;
              } else {
                result[0] += -0.03269283219597131;
              }
            } else {
              result[0] += 0.01064951676333565;
            }
          } else {
            result[0] += 0.0008431897702987475;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
        result[0] += 0.0018995370344808277;
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
          result[0] += -0.007412837282817768;
        } else {
          result[0] += 0.027875828026924433;
        }
      }
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)10.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
              result[0] += -0.10019079768768124;
            } else {
              result[0] += 0.07097628480415673;
            }
          } else {
            result[0] += -0.0060320587353048525;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
            result[0] += -0.017355599665043915;
          } else {
            result[0] += 0.026978501242467864;
          }
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.10687182702723215;
              } else {
                result[0] += -0.017669776389540683;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += 0.018034258343918207;
                  } else {
                    result[0] += -0.02258405031897669;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.002577838775780504;
                    } else {
                      result[0] += -0.02267929020397114;
                    }
                  } else {
                    result[0] += 0.0014785066579025437;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.01784198129246906;
                  } else {
                    result[0] += 0.05471529679594037;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.087577104568482333) ) ) {
                    result[0] += -0.03987219213121118;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.549068689346314365) ) ) {
                        result[0] += -0.01492206578765022;
                      } else {
                        result[0] += 0.05739982143197695;
                      }
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.07560352022255723;
                      } else {
                        result[0] += -0.031317018323900304;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.053887876776412935;
          }
        } else {
          result[0] += 9.699610669880478e-05;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                result[0] += 0.00492002556352291;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.009323446035882248;
                  } else {
                    result[0] += 0.02862846699746094;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
                    result[0] += -0.06940823516589739;
                  } else {
                    result[0] += -0.009042054923578752;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.890938758850098544) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
                      result[0] += 0.015800270399370726;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                        result[0] += 0.026052713642945688;
                      } else {
                        result[0] += -0.01266079282734598;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += 0.05873364267771705;
                    } else {
                      result[0] += -0.0456416308785479;
                    }
                  }
                } else {
                  result[0] += 0.03521257811998188;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.05363449723497874;
                } else {
                  result[0] += 0.0021525098386525913;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.08125248029235499;
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.821652412414551669) ) ) {
                    result[0] += -0.007768006612179257;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
                      result[0] += 0.08699366002445696;
                    } else {
                      result[0] += -0.012130105355912787;
                    }
                  }
                }
              } else {
                result[0] += -5.8302175008063026e-06;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.453179836273194248) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                  result[0] += 0.008665908936529897;
                } else {
                  result[0] += 0.050456810488506934;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += 0.05865812781120253;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                      result[0] += 0.04257246372714271;
                    } else {
                      result[0] += -0.047213045332070984;
                    }
                  }
                } else {
                  result[0] += 0.06854598046942213;
                }
              }
            }
          }
        } else {
          result[0] += -0.01823755945186872;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
                result[0] += -0.04288651595725081;
              } else {
                result[0] += 0.027947643134524393;
              }
            } else {
              result[0] += 0.08822016341523843;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                result[0] += 0.011113704852861292;
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.173316955566407138) ) ) {
                    result[0] += -0.041289533785956095;
                  } else {
                    result[0] += 0.06945844287176732;
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.64700984954834162) ) ) {
                    result[0] += -0.016914934821687846;
                  } else {
                    result[0] += 0.023889543360771072;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.034704228339079546;
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += 0.011556953426643379;
                } else {
                  result[0] += 0.17448974068951179;
                }
              }
            }
          }
        } else {
          result[0] += 0.15307477776457537;
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
        result[0] += 0.10558924305562008;
      } else {
        result[0] += 0.008766184529027348;
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
            result[0] += 0.04431196750916669;
          } else {
            result[0] += 0.004777350307548495;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81109094619751154) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.016880514893078884;
              } else {
                result[0] += -0.0030218975473899437;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.005172056915529205;
              } else {
                result[0] += 0.06401034884309102;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.07920817532574645;
              } else {
                result[0] += 0.013339061280585441;
              }
            } else {
              result[0] += -0.006087507881603525;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.0496513960749072;
            } else {
              result[0] += 0.004351388732304181;
            }
          } else {
            result[0] += 0.01686501938834951;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.005711555821624012;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.01987270292180731;
                } else {
                  result[0] += 0.005395277514901559;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.052422395298486635;
                } else {
                  result[0] += -0.01763001346949138;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.025553271586232485;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.07273849055116348;
                } else {
                  result[0] += 0.04133425414505014;
                }
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += -0.018151309673876247;
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.015405688361759257;
                    } else {
                      result[0] += -0.0731313869998556;
                    }
                  } else {
                    result[0] += -0.002033551657295887;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.08320000204896841;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.03012857864776899;
                  } else {
                    result[0] += -0.021029757728759216;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -4.7481282868863114e-05;
    }
  } else {
    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.499747991561890537) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
          result[0] += -0.13235687953776118;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
            result[0] += 0.01434235083504901;
          } else {
            result[0] += -0.04079550239274743;
          }
        }
      } else {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.07076667887060985;
          } else {
            result[0] += 0.0077872799653623944;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
            result[0] += 0.03598875233193973;
          } else {
            result[0] += -0.008356540816525062;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
          result[0] += -0.004780557980004132;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                    result[0] += -0.0006487076553032969;
                  } else {
                    result[0] += 0.05618090829796191;
                  }
                } else {
                  result[0] += 0.03343652982541709;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.249904870986938921) ) ) {
                      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.11286551636161833;
                      } else {
                        result[0] += -0.028108232306581594;
                      }
                    } else {
                      result[0] += -0.024966983381360462;
                    }
                  } else {
                    result[0] += 0.00987329777056973;
                  }
                } else {
                  result[0] += 0.018064900161041078;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
                  result[0] += 0.007044875652605887;
                } else {
                  result[0] += 0.04010548695056873;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.837148427963257724) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                      result[0] += 0.0025306903343561907;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.022538185119630683) ) ) {
                          result[0] += -0.01545601711908004;
                        } else {
                          result[0] += 0.08327389068299963;
                        }
                      } else {
                        result[0] += -0.05974658630260446;
                      }
                    }
                  } else {
                    result[0] += 0.04059500488194525;
                  }
                } else {
                  result[0] += -0.10770326868652347;
                }
              }
            }
          } else {
            result[0] += -0.025534333614053745;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
          if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
              result[0] += 0.05178220617150109;
            } else {
              result[0] += -0.043703107140406726;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
                result[0] += 0.005901402899611976;
              } else {
                result[0] += 0.05097945491481916;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += 0.021150785371345737;
                } else {
                  result[0] += 0.09049076593578113;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.08312442382175786;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                    result[0] += 0.009537250607825292;
                  } else {
                    result[0] += -0.02510255022697614;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.08884593249174377;
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00027297009249825107;
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.0683107750049143;
          } else {
            result[0] += -0.019097966848229;
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.06513438027884652;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.231051445007325107) ) ) {
                          result[0] += -0.13234651820456186;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
                            result[0] += 0.014555898772597088;
                          } else {
                            result[0] += -0.07880293111886334;
                          }
                        }
                      } else {
                        result[0] += -0.00496164878919204;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.0301328851186704;
                      } else {
                        result[0] += -0.016123072261111853;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
                      result[0] += 0.10163407086374049;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                        result[0] += 0.055717081623477195;
                      } else {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.714014530181885654) ) ) {
                          result[0] += -0.09038768039183648;
                        } else {
                          result[0] += 0.007169034968700677;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.039425849054482376;
                    } else {
                      result[0] += 0.08114119108446423;
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.02134705134833423;
                    } else {
                      result[0] += -0.09835145052406946;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.007766782564201143;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                        result[0] += -0.012495749411869364;
                      } else {
                        result[0] += 0.10682606685040885;
                      }
                    } else {
                      result[0] += -0.02976612357729145;
                    }
                  } else {
                    result[0] += -0.043447284570666246;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                result[0] += 0.010027239226964635;
              } else {
                result[0] += -0.012015567747381427;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.668367385864259589) ) ) {
                    result[0] += -0.1021038933120698;
                  } else {
                    result[0] += 0.02551543365127539;
                  }
                } else {
                  result[0] += 0.06024282985794104;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                  result[0] += -0.05641448868079688;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                    result[0] += 0.01282040980944093;
                  } else {
                    result[0] += -0.0317370252862652;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.230558872222901279) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.473471879959107333) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                        result[0] += -0.0033023097589425323;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.597218394279480425) ) ) {
                            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                                result[0] += -0.004741008491613829;
                              } else {
                                result[0] += 0.05445877871827553;
                              }
                            } else {
                              result[0] += -0.02624468487850892;
                            }
                          } else {
                            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                              result[0] += 0.024539202643526454;
                            } else {
                              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.276966691017151323) ) ) {
                                result[0] += 0.004231449759836077;
                              } else {
                                result[0] += 0.05227111073694233;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.01962013430853432;
                        }
                      }
                    } else {
                      result[0] += -0.005587686269821899;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                        result[0] += -0.01366971577620357;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                          result[0] += 0.007016418361795291;
                        } else {
                          result[0] += 0.09955646945574792;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                          result[0] += 0.01976747359405838;
                        } else {
                          result[0] += 0.07960571695717418;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                          result[0] += 0.01101877001252332;
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.83939445018768355) ) ) {
                            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.022903277196445825;
                            } else {
                              result[0] += -0.11112786647360198;
                            }
                          } else {
                            result[0] += 0.14039098050000828;
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.09643409240250243;
                }
              } else {
                result[0] += -0.05393528687723408;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                result[0] += -7.507313233013801e-05;
              } else {
                result[0] += -0.07730405754801918;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
              result[0] += 0.020861235329137248;
            } else {
              result[0] += -0.04069396069760707;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += -0.009966988716237584;
              } else {
                if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.01699161254280513;
                } else {
                  if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.07197413717033518;
                  } else {
                    result[0] += 0.03713271836660794;
                  }
                }
              }
            } else {
              result[0] += -0.02480299421042031;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.021275585283109296;
            } else {
              result[0] += 0.0912754017725784;
            }
          }
        }
      }
    } else {
      result[0] += 0.07190516251854816;
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -0.0004208559057604865;
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.008485678927878968;
            } else {
              result[0] += 0.012913077477493935;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.067782521247864214) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.016810894012452948) ) ) {
                  result[0] += -0.04387336337827641;
                } else {
                  result[0] += 0.019833450082007948;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    result[0] += 0.06140589601050205;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                      result[0] += -0.07084139178738885;
                    } else {
                      result[0] += 0.04350552450637826;
                    }
                  }
                } else {
                  result[0] += 0.01946363832166346;
                }
              }
            } else {
              result[0] += -0.04384510949400522;
            }
          }
        } else {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += 0.005291134956197682;
            } else {
              result[0] += -0.07017788492974;
            }
          } else {
            result[0] += -0.06822210309587746;
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.08917404034274437;
            } else {
              result[0] += 0.005365107298929438;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.09605199866420083;
              } else {
                result[0] += -0.01826465896922184;
              }
            } else {
              result[0] += -0.004092845896377213;
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.023302852865331264;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.007584507876783874;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                  result[0] += 0.002222215379813106;
                } else {
                  result[0] += 0.025547632686512098;
                }
              }
            }
          } else {
            result[0] += -0.08949752894604267;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += -0.06410881289381001;
        } else {
          result[0] += -0.015432164699997156;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                    result[0] += 0.05636516680946039;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
                      result[0] += -0.0725016190057772;
                    } else {
                      result[0] += -0.011218372631959928;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += 0.06345132248399198;
                  } else {
                    result[0] += 0.0028846683354660924;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                  result[0] += 0.053633957521507486;
                } else {
                  result[0] += -0.007346877424149748;
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0307033988273368;
                } else {
                  result[0] += 0.0700505227845983;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.09627618958230276;
                } else {
                  result[0] += 0.01595637967600027;
                }
              }
            }
          } else {
            result[0] += -0.0009685705517985011;
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.303973913192749912) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.338562726974488193) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04909475331200841;
                  } else {
                    result[0] += 0.06388535004719144;
                  }
                } else {
                  result[0] += 0.10011387922364537;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.034065099974764944;
                } else {
                  result[0] += -0.06844459370251596;
                }
              }
            } else {
              result[0] += -0.04251340584038659;
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.436733961105347568) ) ) {
              result[0] += -0.0022477763162045846;
            } else {
              result[0] += 0.024298160893219994;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.825982809066773349) ) ) {
          result[0] += -0.003608048482251949;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.970040798187256748) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.00627779473643755;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0024856951271016356;
                    } else {
                      result[0] += 0.07341001921054573;
                    }
                  } else {
                    result[0] += 0.029988737666506206;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.333392620086671698) ) ) {
                    result[0] += 0.005661053289527717;
                  } else {
                    result[0] += 0.06877042140856048;
                  }
                }
              }
            } else {
              result[0] += -0.05497832108747144;
            }
          } else {
            result[0] += -0.019805275228297448;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
              result[0] += 0.05443768299939428;
            } else {
              result[0] += -0.025977940254993937;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.026655379644396338;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
                result[0] += -0.012960491680671185;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += 0.10713854751560477;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                      result[0] += 0.04147554867484742;
                    } else {
                      result[0] += 0.005417187043007183;
                    }
                  } else {
                    result[0] += -0.07528750873811359;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.037367737960626854;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.611996650695801669) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.32014131546020685) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
            result[0] += 0.0066374957415174865;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += 0.006293648090636735;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08547641365013539;
              } else {
                result[0] += -0.018749011668092435;
              }
            }
          }
        } else {
          result[0] += 0.019324784271672273;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += 0.02434857924493155;
          } else {
            result[0] += -0.0819591564956686;
          }
        } else {
          result[0] += -0.018712384504779193;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
        result[0] += 0.01305949847699818;
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.170116901397705966) ) ) {
            result[0] += -0.015900837413297987;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += -0.06345364901134988;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                result[0] += -0.008224687129265445;
              } else {
                result[0] += -0.06322870466463898;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
            result[0] += 0.01908937491541178;
          } else {
            result[0] += -0.014275595679672643;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
          result[0] += -0.0004191147436613951;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
            result[0] += -0.03744486813663012;
          } else {
            result[0] += -0.0062779545856862145;
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.0015923396956295328;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.06641818770397749;
            } else {
              result[0] += -0.01674327815738608;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.802901029586792436) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.53886294364929288) ) ) {
                result[0] += -0.029054593170715348;
              } else {
                result[0] += -0.0034872793546145957;
              }
            } else {
              result[0] += -0.08927228176675284;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += -0.0012867124082263732;
        } else {
          result[0] += -0.03189860356361562;
        }
      } else {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
            result[0] += 0.0032578477555197725;
          } else {
            result[0] += -0.038282824288766316;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.62696647644043146) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.022124890577279555;
                    } else {
                      result[0] += 0.001801532734099818;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.00737250613991099;
                      } else {
                        result[0] += -0.017769346090633704;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.04682663007282234;
                        } else {
                          result[0] += -0.013574366781584819;
                        }
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            result[0] += 0.11333276689762849;
                          } else {
                            result[0] += 0.0016523155015312559;
                          }
                        } else {
                          result[0] += -0.04170731537339109;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.01873078716262562;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += -0.01414648794984835;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.008177937145824248;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.0037475560272130326;
                        } else {
                          result[0] += -0.0661782164364785;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += -0.004311866131311079;
                  } else {
                    result[0] += -0.04497213907571575;
                  }
                } else {
                  result[0] += 0.0013346458709296274;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                  result[0] += 0.004191189870351762;
                } else {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.008242977681937703;
                      } else {
                        result[0] += -0.07833759724165879;
                      }
                    } else {
                      result[0] += 0.00533242890706728;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.777633190155030185) ) ) {
                      result[0] += -0.004581166033905177;
                    } else {
                      result[0] += -0.0649195210562006;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.837148427963257724) ) ) {
                    result[0] += -0.004851752543663521;
                  } else {
                    result[0] += 0.03678312907013611;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.01840442291962218;
                      } else {
                        result[0] += 0.0645963399750039;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                        result[0] += -0.039867555791035236;
                      } else {
                        result[0] += 0.008675368113616177;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.039331421967591644;
                    } else {
                      result[0] += 0.006631826048115772;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0021156173432426146;
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.015836466006095392;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.02581628467532267;
                } else {
                  result[0] += 0.0841295869445576;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
      result[0] += -0.00018056304657797174;
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.98865938186645685) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                result[0] += -0.09190328726562169;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
                  result[0] += -0.014316855273314819;
                } else {
                  result[0] += 0.10009563770167224;
                }
              }
            } else {
              result[0] += -0.08614433213107164;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.0004916692316041949;
              } else {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                    result[0] += -0.051255179680421065;
                  } else {
                    result[0] += -0.017247733210965583;
                  }
                } else {
                  result[0] += 0.07336607189064416;
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.913499355316162998) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                  result[0] += -0.017795498687612033;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.05743390220379771;
                  } else {
                    result[0] += -0.11161018209679052;
                  }
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.048149813466193805;
                  } else {
                    result[0] += -0.08335791532261055;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                    result[0] += -0.032037896314805685;
                  } else {
                    result[0] += 0.07627974152752294;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.029989081762835386;
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
          result[0] += -0.011243700515871459;
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.010744391596572776;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.1436223353150332;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.29014396667480646) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.07693735262896037;
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                    result[0] += 0.0010800761796513692;
                  } else {
                    result[0] += 0.13741975487541955;
                  }
                }
              } else {
                result[0] += 0.1885393313695953;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
          result[0] += -0.02535922478134492;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
              result[0] += 0.043649273027291824;
            } else {
              result[0] += 0.003966561732351408;
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += -0.02814881739805349;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.036931514739991123) ) ) {
                    result[0] += 0.09109464249613734;
                  } else {
                    result[0] += 0.010433183717241242;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.003936362221910235;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                    result[0] += -0.02590981993083379;
                  } else {
                    result[0] += -0.08004171172610336;
                  }
                }
              }
            } else {
              result[0] += -0.07792754830643768;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.134879350662232333) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
              result[0] += -0.0062236729026881;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.010111753841151094;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.790659427642823154) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.003359535579511648;
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.04499086277169197;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += 0.025289945417087362;
                            } else {
                              result[0] += -0.08944820998649555;
                            }
                          } else {
                            result[0] += -0.1128058411511913;
                          }
                        }
                      } else {
                        result[0] += 0.045017255736173815;
                      }
                    }
                  } else {
                    result[0] += -0.020835167878183153;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.023400783538819248) ) ) {
                      result[0] += -0.06621159392898902;
                    } else {
                      result[0] += -0.01903486054892587;
                    }
                  } else {
                    result[0] += 0.008454323756606555;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              result[0] += 0.016290766876671667;
            } else {
              result[0] += -0.05155473345337907;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.76642942428588956) ) ) {
                result[0] += 0.006062215967463375;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += 0.1646728175036489;
                } else {
                  result[0] += 0.016491412649508066;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                result[0] += 0.1074721575462157;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.05252143253940022;
                        } else {
                          result[0] += -0.0073132280745665445;
                        }
                      } else {
                        result[0] += 0.0753247948067804;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                          result[0] += 0.07111958184111197;
                        } else {
                          result[0] += 0.016298377292488446;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.847910165786744052) ) ) {
                          result[0] += -0.03046576988149359;
                        } else {
                          result[0] += -0.12327107549546948;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04564743676695182;
                  }
                } else {
                  result[0] += -0.03220094568283532;
                }
              }
            }
          } else {
            result[0] += -0.031275570420735456;
          }
        }
      }
    } else {
      result[0] += 0.05926292112015156;
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.0883522033691424) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
              result[0] += 0.007594973302955688;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.005149706957841687;
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                    result[0] += 0.001588368413075091;
                  } else {
                    result[0] += -0.031104601429752785;
                  }
                } else {
                  result[0] += -0.03496674515786063;
                }
              }
            }
          } else {
            result[0] += 0.0213755361153038;
          }
        } else {
          result[0] += -0.02180240726062249;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += 0.008597897609709434;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.338562726974488193) ) ) {
                result[0] += -0.005684669539883517;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.06900365819388173;
                } else {
                  result[0] += 0.007893938123824319;
                }
              }
            }
          } else {
            result[0] += 0.008068571900960747;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.807895898818970615) ) ) {
              result[0] += -0.022236580466279767;
            } else {
              result[0] += -0.055571720429186566;
            }
          } else {
            result[0] += 0.0013378487772217498;
          }
        }
      }
    } else {
      result[0] += -0.06923384322697355;
    }
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.0007533407761810809;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.333273410797120029) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.003252953739169371;
            } else {
              result[0] += -0.038013823042037924;
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
              result[0] += -0.05261059763749078;
            } else {
              result[0] += 0.013754167581041261;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.014537539603502389;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.85305833816528498) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.01108069193793645;
                } else {
                  result[0] += -0.058192251129093354;
                }
              } else {
                result[0] += 0.00719689245426035;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.0031437444026128124;
                    } else {
                      result[0] += 0.04140276894491239;
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.04658686471590586;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.138333082199097124) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.007230519167754425;
                        } else {
                          result[0] += 0.0448008703346405;
                        }
                      } else {
                        result[0] += -0.05357744955602817;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.07293939590454279) ) ) {
                    result[0] += -0.02242449533900474;
                  } else {
                    result[0] += 0.05855165259000329;
                  }
                }
              } else {
                result[0] += -0.01909745649534084;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += -0.0009365255320687012;
        } else {
          result[0] += -0.030014033876058884;
        }
      } else {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
            result[0] += 0.0031329123124740547;
          } else {
            result[0] += -0.036656152455800416;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
                  result[0] += 0.006454058077015013;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.499747991561890537) ) ) {
                    result[0] += 0.01487051823815527;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += -0.05413758681922124;
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.006357122827254149;
                      } else {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.04207241295429548;
                        } else {
                          result[0] += -0.013391768989649855;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                    result[0] += -0.04078073698910466;
                  } else {
                    result[0] += 0.057887248775432554;
                  }
                } else {
                  result[0] += 0.00391864699358699;
                }
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 0.04032941542791242;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.236541748046876776) ) ) {
                    result[0] += -0.05877303279585114;
                  } else {
                    result[0] += 0.09126375524595902;
                  }
                }
              } else {
                result[0] += -0.001088923013860558;
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.00676276757425633;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11192369461059748) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.799612998962403232) ) ) {
                    result[0] += -0.022901384515151578;
                  } else {
                    result[0] += 0.06341937346754714;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.02191724477581604;
                    } else {
                      result[0] += 0.07176854729128133;
                    }
                  } else {
                    result[0] += 0.009607024670531435;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.017773883561760476;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.01020168856457137;
                  } else {
                    result[0] += 0.05920068340105553;
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                      result[0] += -0.0052474709001916;
                    } else {
                      result[0] += -0.05535423843610987;
                    }
                  } else {
                    result[0] += 0.059887330349232784;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[35].missing != -1) && (data[35].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.490982532501221591) ) ) {
      result[0] += 0.004130873126671605;
    } else {
      result[0] += 0.033239354898346084;
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.463808774948121005) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.015259588211627923;
                } else {
                  result[0] += 0.09068171333517286;
                }
              } else {
                result[0] += 0.005699676032658961;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.023854714578484003;
                } else {
                  result[0] += 0.10090129690222105;
                }
              } else {
                result[0] += -0.022642041549898462;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.014966163196779278;
                } else {
                  result[0] += -0.0022976710102115823;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)7.006656408309937412) ) ) {
                      result[0] += 0.019894800402741514;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                        result[0] += 0.12059182398048024;
                      } else {
                        result[0] += -0.08401423194926694;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.030543989288149817;
                        } else {
                          result[0] += -0.15061155322031064;
                        }
                      } else {
                        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.14935137208833713;
                        } else {
                          result[0] += -0.06860190249615522;
                        }
                      }
                    } else {
                      result[0] += 0.008078643243837842;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += -0.015928857353523822;
                  } else {
                    result[0] += 0.09801236050291307;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.012666931690858358;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.0011809299258950857;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.022382364144570982;
                  } else {
                    result[0] += 0.01861116169268801;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.021324520203498606;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                    result[0] += -0.011281570571237725;
                  } else {
                    result[0] += 0.029101405397790915;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += 0.026656449150554918;
                    } else {
                      result[0] += 0.00352358414296719;
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.05031342135578801;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.55507802963257014) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
                            result[0] += 0.1012993185906163;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                              result[0] += -0.1579740051308219;
                            } else {
                              result[0] += 0.0085625068758337;
                            }
                          }
                        } else {
                          result[0] += -0.07363823703515729;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.06034901348881693;
                        } else {
                          result[0] += -0.1322339212845409;
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += 0.019229833660314426;
                        } else {
                          result[0] += -0.03651966117918578;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.015711758825611447;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.0018480113176469228;
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += -0.07993304312753641;
                        } else {
                          result[0] += 0.07566635624692775;
                        }
                      } else {
                        result[0] += -0.27744168285786913;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += 0.1699461394946866;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.465945959091187412) ) ) {
                        result[0] += -0.0002111121041491659;
                      } else {
                        result[0] += 0.13463022592172608;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.03026170603175207;
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.027909290444339138;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.587220668792725498) ) ) {
                result[0] += 0.019017519370315303;
              } else {
                result[0] += -0.02929042740720546;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
          result[0] += 0.00013812522516436134;
        } else {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += 0.016527490521092873;
          } else {
            result[0] += -0.049622891978099895;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.0018459062471203017;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.400584220886231357) ) ) {
            result[0] += -0.01069950001378163;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
                  result[0] += -0.018522537793707525;
                } else {
                  result[0] += 0.07797209818409445;
                }
              } else {
                result[0] += 0.09074613108118154;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.109050035476685458) ) ) {
                result[0] += 0.003405853003206766;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                  result[0] += 0.016893475294998253;
                } else {
                  result[0] += -0.028003398611155663;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.01679450923769688;
          } else {
            result[0] += 0.059368845451150655;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[34].missing != -1) && (data[34].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.490982532501221591) ) ) {
      result[0] += 0.0039705925445460205;
    } else {
      result[0] += 0.03146836390585012;
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.0009927598948808786;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -4.382281542678353e-05;
            } else {
              result[0] += 0.003050896098701401;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.08194107305735429;
            } else {
              result[0] += 0.012093288284085409;
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.11900938927275745;
              } else {
                result[0] += -0.00997887967917384;
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.027667367416347035;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += 0.0684153619032392;
                      } else {
                        result[0] += -0.00036283528674498425;
                      }
                    } else {
                      if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.970040798187256748) ) ) {
                            result[0] += 0.01763288998993569;
                          } else {
                            result[0] += -0.08643859305439221;
                          }
                        } else {
                          result[0] += -0.020620334003730513;
                        }
                      } else {
                        result[0] += 0.004531636095401181;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.06829310802878467;
                    } else {
                      result[0] += 0.0007207818404481565;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.138696432113648349) ) ) {
                      result[0] += 0.0034139049751514793;
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.05707675068802292;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                            result[0] += 0.027759590324632023;
                          } else {
                            result[0] += -0.013464000351529956;
                          }
                        }
                      } else {
                        result[0] += 0.01937664581158575;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.027653868250147734;
                    } else {
                      result[0] += -0.0023807793685705902;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
                      result[0] += 0.013491206506278101;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                        result[0] += 0.034553955002022484;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += -0.03741751471946551;
                        } else {
                          result[0] += 0.034146832982335805;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.02193641662597834) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.169590950012207919) ) ) {
                        result[0] += 0.006397637138709414;
                      } else {
                        result[0] += -0.03082589986745884;
                      }
                    } else {
                      result[0] += -0.03523103183298445;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.02050847630858797;
                        } else {
                          result[0] += 0.009095051810159504;
                        }
                      } else {
                        result[0] += -0.05417961452883328;
                      }
                    } else {
                      result[0] += 0.010982569787396227;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.010630726156122088;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.00420028217326805;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.016186055486621127;
                    } else {
                      result[0] += 0.027104361452053062;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                      result[0] += -0.012599218098371287;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                        result[0] += -0.030315449943283664;
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.250939607620240146) ) ) {
                          result[0] += 0.05999311731749405;
                        } else {
                          result[0] += 0.01086356847809126;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.024739417718283172;
                  } else {
                    result[0] += 0.02276071701918322;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.02353883076559905;
                  } else {
                    result[0] += 0.010784207809973231;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.0017905769285268777;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.026820278282611215;
            } else {
              result[0] += 0.0006943543302435102;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
              result[0] += 0.06960423397752488;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += -0.01108968642711766;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.004791665746882195;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                        result[0] += 0.021985942363368816;
                      } else {
                        result[0] += 0.08339240675610703;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.44140100479126021) ) ) {
                        result[0] += 0.017359826504666945;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.308072090148926669) ) ) {
                          result[0] += -0.0002846239413867581;
                        } else {
                          result[0] += -0.03628645926116144;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.032151846704556696;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.017622741840175413;
                  } else {
                    result[0] += 0.17953346856234814;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.017641503372516156;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.685236454010010654) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.490982532501221591) ) ) {
        result[0] += 0.003570674398794354;
      } else {
        result[0] += 0.02711580164590049;
      }
    } else {
      result[0] += 0.12704439308545587;
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.00019059533998843427;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.382196187973023349) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                result[0] += 0.017560747666708152;
              } else {
                result[0] += -0.012720324758614569;
              }
            } else {
              result[0] += 0.01698728859399555;
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.02720612770720752;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.019517957893217877;
                } else {
                  result[0] += -0.09886128675378981;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.827628135681153232) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.020127415657043901) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.594409704208374912) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.0028666589904297116;
                        } else {
                          result[0] += -0.020224705121040184;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
                          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.014243124728648235;
                          } else {
                            result[0] += 0.0847149728626359;
                          }
                        } else {
                          result[0] += 0.011234466221440535;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.06966042926390269;
                        } else {
                          result[0] += 0.012332520578494807;
                        }
                      } else {
                        result[0] += 0.023774371489256003;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                      result[0] += -0.0788779442103951;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.97887301445007413) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                            result[0] += -0.07524257061394302;
                          } else {
                            result[0] += 0.014116331284604168;
                          }
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.990212917327881748) ) ) {
                              result[0] += 0.06171690452059837;
                            } else {
                              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.053540233228474114;
                              } else {
                                result[0] += -0.07581042082910797;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                              result[0] += -0.08251849069388772;
                            } else {
                              result[0] += 0.020448608983394578;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                            result[0] += 0.014755851672975953;
                          } else {
                            result[0] += -0.07470756979811267;
                          }
                        } else {
                          result[0] += -0.0936310964400601;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.036362978402364375;
                }
              } else {
                result[0] += 0.04354146113445445;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.98865938186645685) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
              result[0] += -0.013587367905409776;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.12857069269137641;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.422362327575684482) ) ) {
                  result[0] += 0.021318793106949906;
                } else {
                  result[0] += 0.17070668130576222;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += -0.09459845901617747;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                      result[0] += 0.0010147764424364126;
                    } else {
                      result[0] += 0.12772068063976053;
                    }
                  }
                } else {
                  result[0] += -0.0290284903703678;
                }
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.026151984264566802;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81435346603393732) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.019229173660279208) ) ) {
                      result[0] += -0.0024504501905387253;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12508964538574396) ) ) {
                        result[0] += 0.1194217593819531;
                      } else {
                        result[0] += -0.06842150755301105;
                      }
                    }
                  } else {
                    result[0] += 0.07345687524500005;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08033169662469444;
              } else {
                result[0] += -0.030046372597580164;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.060294389724732333) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.08015770642069639;
            } else {
              result[0] += -0.026785931658157347;
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                  result[0] += 0.011076855144360428;
                } else {
                  result[0] += 0.14122886149573383;
                }
              } else {
                result[0] += -0.05877738305518124;
              }
            } else {
              if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.035870845114589274;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                  result[0] += 0.03516139135190621;
                } else {
                  result[0] += 0.1389403873772552;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
          result[0] += -0.018896235399606987;
        } else {
          result[0] += -0.00030248619963277345;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.57691621780395685) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
              result[0] += 0.002853672306635507;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                result[0] += 0.014838396978301803;
              } else {
                result[0] += -0.028301834127777654;
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
                result[0] += 0.03945208760355037;
              } else {
                result[0] += 0.008581607219402798;
              }
            } else {
              result[0] += 0.053374411485285815;
            }
          }
        } else {
          result[0] += -0.03302337601879175;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.014798719477957176;
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                result[0] += 0.015255555066648208;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                  result[0] += -0.009319985428437837;
                } else {
                  result[0] += -0.054714493860249784;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                result[0] += 0.0065419906165249415;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += 0.027269294516103555;
                } else {
                  result[0] += -0.03390255114955226;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += -0.025831635293390866;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.009886410640776257;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                      result[0] += 0.05786769734556049;
                    } else {
                      result[0] += 0.02663935664108832;
                    }
                  }
                } else {
                  result[0] += 0.009756870241908175;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.080420010669713;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                  result[0] += -0.019026040392704148;
                } else {
                  result[0] += 0.012885031663272019;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
              result[0] += -0.005508643718593992;
            } else {
              result[0] += -0.014665078108460057;
            }
          } else {
            result[0] += 0.002962884600249111;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.440248489379883701) ) ) {
                result[0] += -0.014088641412495467;
              } else {
                result[0] += -0.0752720592413659;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.023216741123024017;
                } else {
                  result[0] += 0.02318741690657165;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0030783879233112015;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.018351978689581994;
                        } else {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.021719334741163766;
                          } else {
                            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.004178514628655909;
                            } else {
                              result[0] += 0.034922697770421625;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += 0.023231184930832677;
                    }
                  } else {
                    result[0] += 0.04031114201971408;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.611996650695801669) ) ) {
                      result[0] += 0.0004238118804606038;
                    } else {
                      result[0] += 0.03669607965214539;
                    }
                  } else {
                    result[0] += 0.07602064021110533;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += 0.00064535182039443;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.036653881876281165;
                } else {
                  result[0] += -0.029196103101235535;
                }
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                      result[0] += 0.023355885717777795;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                        result[0] += 0.023567771786548328;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.055826002389062004;
                        } else {
                          result[0] += 0.008650677319796712;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.0047937233075586;
                      } else {
                        result[0] += -0.09922921066568804;
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.11554368709100565;
                      } else {
                        result[0] += -0.033025084019734356;
                      }
                    }
                  }
                } else {
                  result[0] += -0.1020492092794748;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
                    result[0] += 0.04323915346254017;
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0057418481929039185;
                    } else {
                      result[0] += 0.09335325243765834;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.05936129556823815;
                    } else {
                      result[0] += -0.02056584047512964;
                    }
                  } else {
                    result[0] += -0.09194990860787185;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.03181906485335807;
              } else {
                result[0] += -0.0016221152597266778;
              }
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.1152043720825292;
                } else {
                  result[0] += -0.026818698651240924;
                }
              } else {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.05291158313730742;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
                      result[0] += -0.006678089294371034;
                    } else {
                      result[0] += 0.012246774499801936;
                    }
                  } else {
                    result[0] += -0.03157795437099919;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += -0.014232921351403022;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                  result[0] += -0.03083052839686399;
                } else {
                  result[0] += 0.0607543225647204;
                }
              } else {
                result[0] += -0.010010469395818488;
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.000638240289535004;
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.00042120793947596067;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.043341875076294833) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += 0.09206150053478768;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.023833627686256673;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.010277061619699032;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                          result[0] += -0.006407784230825906;
                        } else {
                          result[0] += 0.03943607034161178;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.10010089374002831;
                  } else {
                    result[0] += 0.0037563757122125938;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
                    result[0] += 0.045776844733675714;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
                      result[0] += -0.06817932679691205;
                    } else {
                      result[0] += 0.001910302541681677;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.09603050761572256;
                      } else {
                        result[0] += -0.03160100278022961;
                      }
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          result[0] += 0.07587243602248481;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                            result[0] += -0.04115600169013056;
                          } else {
                            result[0] += 0.012102856565553252;
                          }
                        }
                      } else {
                        result[0] += -0.038196391383477496;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += -0.02663091933551927;
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.008300962392492174;
                        } else {
                          result[0] += -0.07614679903377343;
                        }
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                          result[0] += 8.533576228175448e-05;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                            result[0] += -0.06567329798171459;
                          } else {
                            result[0] += 0.06411624165836927;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.0365119914955797;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.022538185119630683) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.020127415657043901) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.06260245549216027;
                  } else {
                    result[0] += 0.0157853502351599;
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.015356542593224538;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.556798219680787021) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                          result[0] += 0.028507288338957527;
                        } else {
                          result[0] += -0.013750049328260003;
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                              result[0] += -0.06867431147942847;
                            } else {
                              result[0] += 0.037316540351469044;
                            }
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.512487888336182529) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                                result[0] += -0.01100048541770944;
                              } else {
                                result[0] += 0.010296065773716629;
                              }
                            } else {
                              result[0] += 0.017702733276188508;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += -0.1133264756581818;
                            } else {
                              result[0] += -0.03305251815208734;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
                                result[0] += -0.060896827607470054;
                              } else {
                                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.241300821304322177) ) ) {
                                  result[0] += 0.02887172109712159;
                                } else {
                                  result[0] += 0.11158499393469737;
                                }
                              }
                            } else {
                              result[0] += -0.015463349619777772;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.901921629905701128) ) ) {
                          result[0] += -0.059274479683857344;
                        } else {
                          result[0] += 0.02058254021960651;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                          result[0] += -0.05546412291260233;
                        } else {
                          result[0] += 0.006364518400723879;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.03456393734024198;
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.12038213780539314;
              } else {
                result[0] += 0.052705887124447286;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += 0.054746758750169655;
          } else {
            result[0] += 0.011324798591477784;
          }
        } else {
          result[0] += -0.09239713320151588;
        }
      }
    } else {
      result[0] += -0.02897245205675661;
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.400584220886231357) ) ) {
          result[0] += -0.008516595624996;
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.006242358313666294;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.004927595316726666;
            } else {
              result[0] += 0.023098184148294027;
            }
          }
        }
      } else {
        result[0] += 0.037571510432803115;
      }
    } else {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
          result[0] += -0.009691089227401086;
        } else {
          result[0] += -0.07954628932444902;
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
          result[0] += 0.007843287888872714;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
            result[0] += 0.005490555124513944;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
              result[0] += -0.0039059857868831726;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.023233135437141846;
              } else {
                result[0] += 0.11526880193686675;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0004066558522181081;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                result[0] += -0.019358177989152892;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.012286429650630136;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.625838756561280185) ) ) {
                      result[0] += -0.010920788987967904;
                    } else {
                      result[0] += 0.05536048023107938;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += 0.05184043523401947;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.04194438325954945;
                            } else {
                              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                                result[0] += -0.016114488596422134;
                              } else {
                                result[0] += -0.09377416375345277;
                              }
                            }
                          } else {
                            result[0] += 0.014858480830542337;
                          }
                        } else {
                          result[0] += -0.06925699436459822;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.459994077682496005) ) ) {
                        result[0] += -0.02970345850934816;
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.695914268493653232) ) ) {
                          result[0] += 0.0685424714451615;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.017167562250412577;
                          } else {
                            result[0] += -0.06481469271096865;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.025591427885519536;
                  }
                }
              }
            } else {
              result[0] += 0.024064172644564907;
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.025349162914803575;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.031236265326351084;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += -0.040481945965095766;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.309778213500978339) ) ) {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.034292634840779095;
                        } else {
                          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.000382235835878834;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                              result[0] += -0.00018781307676055887;
                            } else {
                              result[0] += 0.027137572455047956;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.021687368105269805;
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.664659023284913886) ) ) {
                        result[0] += -0.04956890138818573;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.0592544874681589;
                        } else {
                          result[0] += 0.01947275103457256;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.559772014617921698) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.07650278957348737;
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.235757827758790839) ) ) {
                          result[0] += 0.01948592765757374;
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            result[0] += -0.00367278901713039;
                          } else {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.022948875502185943;
                            } else {
                              result[0] += 0.08055836404985851;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.847910165786744052) ) ) {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.448499202728272373) ) ) {
                              result[0] += 0.021712981311568205;
                            } else {
                              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                result[0] += -0.010654072511087262;
                              } else {
                                result[0] += 0.07022684953487252;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += 0.022146801694880258;
                            } else {
                              result[0] += -0.01705001389700855;
                            }
                          }
                        } else {
                          result[0] += -0.1021435692127445;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.0027788151113434037;
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.10719193752707151;
                            } else {
                              result[0] += 0.005623115051226871;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += 0.022787848985619964;
                            } else {
                              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.318498134613038886) ) ) {
                                result[0] += -0.061948014464012116;
                              } else {
                                result[0] += 0.07496128992851318;
                              }
                            }
                          } else {
                            result[0] += 0.06389443535800651;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.012837769009167789;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.06347508833093639;
                      } else {
                        result[0] += -0.022772161595324213;
                      }
                    }
                  }
                } else {
                  result[0] += 0.08265351135330037;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.01897353999872294;
        } else {
          result[0] += -0.08892960303014326;
        }
      }
    } else {
      result[0] += -0.02567320176976559;
    }
  } else {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.917405366897583452) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.400584220886231357) ) ) {
            result[0] += -0.0074634667008723525;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.00611590885416622;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                result[0] += 0.018380746536084385;
              } else {
                result[0] += -0.16181962890179222;
              }
            }
          }
        } else {
          result[0] += 0.034552777035742985;
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.041162281534430314;
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
            result[0] += 0.007740087635474373;
          } else {
            result[0] += -0.0021666801122658982;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.15100884437561124) ) ) {
        result[0] += -0.033961422418011825;
      } else {
        result[0] += 0.1105082834054304;
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95751476287841975) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
              result[0] += 0.0064801981043215735;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.0043751357577927604;
              } else {
                result[0] += -0.0151970690918392;
              }
            }
          } else {
            result[0] += 0.019938650013880954;
          }
        } else {
          result[0] += -0.01857728119742691;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
          result[0] += 0.0011398879303030664;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += -0.04301460240794634;
              } else {
                result[0] += -0.001556776510556971;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.04784169530316735;
                } else {
                  result[0] += -0.12502871589722472;
                }
              } else {
                result[0] += -0.008702202602979901;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
              result[0] += 0.017780290427958503;
            } else {
              result[0] += -0.011011095788775072;
            }
          }
        }
      }
    } else {
      result[0] += -0.06720675878670862;
    }
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.00044269367221237395;
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.338562726974488193) ) ) {
                  result[0] += -0.017465075430586372;
                } else {
                  result[0] += 0.017537291011809276;
                }
              } else {
                result[0] += -0.06860747307857748;
              }
            } else {
              result[0] += -0.06189642493353448;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.0005907467520000837;
              } else {
                result[0] += -0.016452784358354452;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.018707152627300418;
                    } else {
                      result[0] += 0.08487936740175638;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.04261048453703263;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.845905780792238104) ) ) {
                        result[0] += -0.025647800650230097;
                      } else {
                        result[0] += 0.035753843607165034;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.32868957519531428) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.0068596323358216735;
                      } else {
                        result[0] += -0.01803265726386749;
                      }
                    } else {
                      result[0] += 0.021083832006877027;
                    }
                  } else {
                    result[0] += -0.026921634878943695;
                  }
                }
              } else {
                result[0] += -0.009446602673028402;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                    result[0] += -0.01799586947943919;
                  } else {
                    result[0] += 0.010941549387400492;
                  }
                } else {
                  result[0] += -0.028038546560552137;
                }
              } else {
                result[0] += -0.042313425193093536;
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.015627089699254318;
              } else {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.08607149876899797;
                } else {
                  result[0] += -0.0343174746789675;
                }
              }
            }
          } else {
            result[0] += -0.06549267548663253;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += -0.0013396813615813435;
        } else {
          result[0] += -0.026614765156321114;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.62696647644043146) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.00015832280353828782;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.03805413876694985;
                  } else {
                    result[0] += 0.0033778256946294365;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.020713240844774102;
                  } else {
                    result[0] += -0.017279339317191965;
                  }
                }
              } else {
                result[0] += -0.04799648338797883;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += 0.007051541579957327;
                } else {
                  result[0] += -0.033014551516187846;
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.043351183369495644;
                } else {
                  result[0] += -0.006905110686527136;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.827628135681153232) ) ) {
                  result[0] += -0.0046549913885908815;
                } else {
                  result[0] += 0.031223259306768482;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.016165754021690695;
                    } else {
                      result[0] += 0.05975951166219328;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                      result[0] += -0.036890087450763644;
                    } else {
                      result[0] += 0.006422339482648906;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.03654109330169899;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.016798527659344153;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.006078942100351427;
                      } else {
                        result[0] += 0.0458199742065264;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0036347509615418786;
          } else {
            result[0] += 0.02334338927023369;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.013281920547808402;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.011964713733013513;
                  } else {
                    result[0] += -0.011002824273242052;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.03730898770152467;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += 0.01737320756028526;
                    } else {
                      result[0] += -0.012737897189557987;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
                      result[0] += -0.005644208453231497;
                    } else {
                      result[0] += -0.03882285190496322;
                    }
                  } else {
                    result[0] += -0.06307799372604127;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.01595946664716582;
                  } else {
                    result[0] += -0.027181974797072406;
                  }
                } else {
                  result[0] += -0.03689744904528267;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.847910165786744052) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.378218650817871982) ) ) {
                    result[0] += -0.02707891141595637;
                  } else {
                    result[0] += -0.05939924519156044;
                  }
                } else {
                  result[0] += -0.003741828313165483;
                }
              }
            }
          } else {
            result[0] += 0.0010249445130635461;
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.0003833909169609818;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.012781745327659189;
              } else {
                result[0] += -0.0077357711921247786;
              }
            } else {
              result[0] += 0.01290736473651536;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                    result[0] += 0.010581636142523202;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.06367834674190097;
                    } else {
                      result[0] += 0.0026029240478252606;
                    }
                  }
                } else {
                  result[0] += 0.023873999708685964;
                }
              } else {
                result[0] += -0.050683697861782055;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                  result[0] += -0.03689964552543505;
                } else {
                  result[0] += -0.0024569238181605633;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.007226903047989195;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += -0.022715317439910922;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.010464313867557523;
                    } else {
                      result[0] += -0.029303509642264997;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0065090385059254205;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.02403131008076421;
                    } else {
                      result[0] += 0.0019842167074260466;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.06578326254598948;
                        } else {
                          result[0] += -0.014362634242785596;
                        }
                      } else {
                        result[0] += -0.01529671927078667;
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.02426353203783552;
                      } else {
                        result[0] += 0.022416718663015783;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                    result[0] += -0.026610028612691367;
                  } else {
                    result[0] += 0.011508160061243776;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
                result[0] += 0.01806146707153209;
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                    result[0] += -0.029149315419785844;
                  } else {
                    result[0] += -0.06796406615675536;
                  }
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.059720225598494485;
                    } else {
                      result[0] += -0.002739802775252152;
                    }
                  } else {
                    result[0] += 0.002904399989617055;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
            result[0] += -0.026934109786757445;
          } else {
            result[0] += -0.004930633226029008;
          }
        }
      }
    } else {
      result[0] += -0.02337206759095086;
    }
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -4.875324522540634e-05;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.385823249816895419) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0023213707194335263;
              } else {
                result[0] += 0.011612192873386808;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                result[0] += 0.020821611181135548;
              } else {
                result[0] += -0.032236051408847034;
              }
            }
          } else {
            result[0] += -0.018803184980724745;
          }
        } else {
          result[0] += 0.02144648520817794;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.645421981811524326) ) ) {
          result[0] += -0.006722709173606046;
        } else {
          result[0] += 0.022933409458529112;
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.848108768463135654) ) ) {
            result[0] += -0.09591218643470964;
          } else {
            result[0] += 0.10987427741473949;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.994427680969239169) ) ) {
            result[0] += -0.012355445304155966;
          } else {
            result[0] += -0.08020649780136035;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.932935476303101474) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0028384530892562212;
          } else {
            result[0] += 0.015214895403654139;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.018528242511615677;
            } else {
              result[0] += -0.07768647397075422;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.018613697915736103;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.043341875076294833) ) ) {
                result[0] += 0.019950442722128803;
              } else {
                result[0] += -0.01034505881662256;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
          result[0] += 0.01160253673217175;
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.03630276306699327;
          } else {
            result[0] += -0.006262032956347061;
          }
        }
      }
    } else {
      result[0] += -0.06309098292615493;
    }
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -1.1915900224309748e-06;
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.17202329635620295) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.010533769906378714;
              } else {
                result[0] += -0.04077884539118678;
              }
            } else {
              result[0] += 0.003403518983051332;
            }
          } else {
            result[0] += 0.003234113787810444;
          }
        } else {
          result[0] += -0.017957267152465345;
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.012886469649643961;
            } else {
              result[0] += 0.007171793007884232;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.610357046127320224) ) ) {
                result[0] += -0.006232556495920442;
              } else {
                result[0] += -0.03828469540475939;
              }
            } else {
              result[0] += -0.03506598689498066;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.006067187812255461;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.06929420778257828;
                      } else {
                        result[0] += -0.020172921464687566;
                      }
                    } else {
                      result[0] += -0.01695638364099541;
                    }
                  } else {
                    result[0] += -0.007013743953737105;
                  }
                }
              } else {
                result[0] += -0.0008422301236595067;
              }
            } else {
              result[0] += 0.0072464421356953565;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.022029292953358592;
                } else {
                  result[0] += -0.012962496192532508;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.949854612350464755) ) ) {
                      result[0] += 0.004124896801892584;
                    } else {
                      result[0] += -0.07382665066739418;
                    }
                  } else {
                    result[0] += -0.03856510141574479;
                  }
                } else {
                  result[0] += 0.01274908704922109;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += 0.07156058745060359;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.99033999443054288) ) ) {
                  result[0] += -0.004440972673845045;
                } else {
                  result[0] += 0.021217739347200615;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            result[0] += 0.023279198798418742;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.018527607754096613;
            } else {
              result[0] += -0.04893953114158542;
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
              result[0] += -0.0026663277762331242;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                result[0] += -0.023743061045876455;
              } else {
                result[0] += 0.03542929732236232;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.84331607818603693) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.0031913428862865058;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                        result[0] += -0.00733983813466549;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += -0.020526478665347476;
                        } else {
                          result[0] += -0.06199761029604868;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                        result[0] += 0.010153260923700246;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                            result[0] += -0.05002350331614237;
                          } else {
                            result[0] += 0.03935336249394211;
                          }
                        } else {
                          result[0] += -0.004928306945732954;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.07355236927387157;
                      } else {
                        result[0] += 0.012266482950838726;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.040993875640297925;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                      result[0] += -0.05102300772495038;
                    } else {
                      result[0] += 0.010541494747050559;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12055253982544123) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.005031343379354559;
                  } else {
                    result[0] += -0.0387573416295228;
                  }
                } else {
                  result[0] += 0.0012837480990629654;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += 0.00464112066136264;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0005929776844764854;
                } else {
                  result[0] += 0.0498036726321411;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.02876455374975321;
        } else {
          result[0] += -0.034875418586175225;
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
            result[0] += -0.00716113401948464;
          } else {
            result[0] += -0.06177784900067895;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.897119760513306552) ) ) {
            result[0] += -0.029750618280235747;
          } else {
            result[0] += 0.021973312818004638;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
              result[0] += -0.0022038599376831467;
            } else {
              if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.003927357806314174;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += -0.0012096218462698486;
                  } else {
                    result[0] += 0.024623611474445292;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.013696773367044916;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                        result[0] += 0.08109090703425942;
                      } else {
                        result[0] += -0.0616407962908328;
                      }
                    } else {
                      result[0] += 0.013072620112118908;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                result[0] += -0.0012478538046659174;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.490982532501221591) ) ) {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.05337487380003169;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
                          result[0] += 0.04596313203544376;
                        } else {
                          result[0] += -0.06270908147917859;
                        }
                      } else {
                        result[0] += 0.06146506279157318;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0015307759738595371;
                      } else {
                        result[0] += 0.05592069408365966;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                        result[0] += -0.019431147905236257;
                      } else {
                        result[0] += -0.001005544371830329;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      result[0] += -0.039088864325016945;
                    } else {
                      result[0] += 0.029963019438030505;
                    }
                  } else {
                    result[0] += -0.01751245091645115;
                  }
                }
              }
            } else {
              result[0] += -0.00011385396324153484;
            }
          }
        } else {
          result[0] += 0.0027877234529594566;
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.004782231458425917;
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.76985812187194913) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.020610142066102954;
                  } else {
                    result[0] += 0.05251234521721076;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += 0.007781605400795874;
                    } else {
                      result[0] += -0.06854304772749657;
                    }
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.07859440463988404;
                    } else {
                      result[0] += 0.019271071424380545;
                    }
                  }
                }
              } else {
                result[0] += -0.030901898133358564;
              }
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += 0.009472332677160637;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0012645454547005439;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.05721383573799839;
                      } else {
                        result[0] += -0.012411390050009297;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += 0.025244840755609562;
                      } else {
                        result[0] += -0.12368524890192455;
                      }
                    } else {
                      result[0] += -0.058409899064451966;
                    }
                  } else {
                    result[0] += 0.004911655964576441;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                  result[0] += -0.01975358864258471;
                } else {
                  result[0] += -0.0719241505017317;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.138696432113648349) ) ) {
              result[0] += -0.019087765767726857;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.827628135681153232) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.008464285759038256;
                  } else {
                    result[0] += -0.048237792890888136;
                  }
                } else {
                  result[0] += 0.06950626924757636;
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.014788627624512607) ) ) {
                    result[0] += -0.005256062327351678;
                  } else {
                    result[0] += 0.026281987120888507;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.610357046127320224) ) ) {
                    result[0] += 0.033466942527867485;
                  } else {
                    result[0] += 0.07343860407509666;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
          result[0] += 0.006177445931334855;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            result[0] += 0.014671403584853549;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += 0.010409240010751449;
              } else {
                result[0] += -0.10237016634635165;
              }
            } else {
              result[0] += -0.0242212857523366;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
          result[0] += -0.012572208713127126;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += -0.020455325432142884;
          } else {
            result[0] += 0.017829831167914544;
          }
        }
      }
    } else {
      result[0] += 0.000368126315820358;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.113679647445679599) ) ) {
          result[0] += -0.04585943730694693;
        } else {
          result[0] += 0.12012412556404478;
        }
      } else {
        result[0] += 0.01585667623789979;
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
            result[0] += -0.014510542181701287;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.025716897632415515;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.006322185452841585;
              } else {
                result[0] += -0.1006099282389129;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += -0.017695247758778124;
          } else {
            result[0] += -0.06818950597073047;
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += 0.0007419766501558156;
                  } else {
                    result[0] += 0.04823838835261387;
                  }
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.0009669586184099544;
                  } else {
                    result[0] += -0.016564454931992035;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.0011788078685788979;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.044488230483370084;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.04967627554175052;
                    } else {
                      result[0] += 0.007895967607042344;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.138333082199097124) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.0022745085432190775;
                      } else {
                        result[0] += 0.06260003692369569;
                      }
                    } else {
                      result[0] += 0.14818928084544072;
                    }
                  } else {
                    result[0] += -0.0035271498020579615;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += -0.006095894363280739;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.05093616849773486;
                      } else {
                        result[0] += -0.013909528081701551;
                      }
                    }
                  } else {
                    result[0] += -0.05721855282693267;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.731793165206910068) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.012881373209716837;
                  } else {
                    result[0] += -0.018657217869575242;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.009641826672256283;
                    } else {
                      result[0] += -0.048437940553179;
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.04992319809351598;
                    } else {
                      result[0] += -0.00814818593289577;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.00368964291698635;
            } else {
              result[0] += 0.05155075732175915;
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.07902466336683717;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.04033777833593277;
                } else {
                  result[0] += -0.016084643926459766;
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.02014452544483189;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                  result[0] += -0.05116843923192437;
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.0025297437671029832;
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0014640039675586837;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.166635274887085849) ) ) {
                        result[0] += 0.017773869420368325;
                      } else {
                        result[0] += 0.11209298909399655;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.07791922648069931;
            } else {
              result[0] += -0.015750423338845364;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.10330905326724434;
          } else {
            result[0] += -0.016017659438814096;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0009584214244809457;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.0516594381067533;
                } else {
                  result[0] += -0.009634949367753734;
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.03945987187543937;
                  } else {
                    result[0] += 0.006832749755438479;
                  }
                } else {
                  result[0] += -0.03394402553734071;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.017792947699463216;
              } else {
                result[0] += 0.04800526682301615;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.087577104568482333) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
                  result[0] += -0.015241833980749986;
                } else {
                  result[0] += -0.07869115707925806;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.011486110269990605;
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.06541241280851615;
                      } else {
                        result[0] += -0.06685819817706913;
                      }
                    } else {
                      result[0] += -0.03162224661879749;
                    }
                  }
                } else {
                  result[0] += 0.06853559381232723;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.05107011040934061;
      }
    } else {
      result[0] += 0.0006923714936843018;
    }
  }
  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.021352093118725254;
          } else {
            result[0] += -0.03697669315041464;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
            result[0] += -0.012351444400500459;
          } else {
            result[0] += -0.047675768678353575;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0018307087321964286;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.57691621780395685) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += -0.0022882346542484762;
                } else {
                  result[0] += 0.008439006402204357;
                }
              } else {
                result[0] += 0.02581996042904692;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.06941016408475903;
              } else {
                result[0] += -0.03727318686064883;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += -0.002678964956019086;
                  } else {
                    result[0] += 0.029148214266580236;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.007453239657969005;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.897119760513306552) ) ) {
                        result[0] += -0.0080293278275353;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += -0.08335623432921611;
                        } else {
                          result[0] += -0.03235586884335375;
                        }
                      }
                    } else {
                      result[0] += 0.012097306054450899;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0027353633280249702;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.009478190056508804;
                  } else {
                    result[0] += 0.044235114924627277;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.005080038836563997;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                        result[0] += -0.026570121548439265;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                          result[0] += -0.017007770069407187;
                        } else {
                          result[0] += -0.06436986314425405;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.169590950012207919) ) ) {
                        result[0] += -0.02733980825561426;
                      } else {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.022817751953658677;
                        } else {
                          result[0] += 0.003925350058590724;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                      result[0] += 0.014042949285743518;
                    } else {
                      result[0] += -0.06361431184709689;
                    }
                  }
                }
              } else {
                result[0] += -0.001589095920593529;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.0013922055775817434;
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.039445065805274494;
                      } else {
                        result[0] += -0.025232869558920092;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                          result[0] += 0.06510113621485074;
                        } else {
                          result[0] += 0.007821957743732285;
                        }
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.05475422675504668;
                        } else {
                          result[0] += 0.005315035499699212;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78560066223144709) ) ) {
                      result[0] += -0.0045997546094387115;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.006382049253301227;
                      } else {
                        result[0] += 0.05569067587176622;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    result[0] += 0.015941809718036267;
                  } else {
                    result[0] += -0.09110239871460045;
                  }
                }
              } else {
                result[0] += -0.027311734356741776;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        result[0] += -0.009132698851870312;
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.03451654926060499;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.309873342514038974) ) ) {
                    result[0] += 0.04415821391460751;
                  } else {
                    result[0] += -0.025654766399239755;
                  }
                } else {
                  result[0] += -0.007399202623138524;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += -0.01567682920200921;
                } else {
                  result[0] += 0.018325305961830133;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.05240096867640442;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.551071166992188388) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                  result[0] += 0.007240842814849021;
                } else {
                  result[0] += -0.019433066144132344;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.0592413135375886;
                } else {
                  result[0] += 0.0023249506841151466;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.138696432113648349) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.310776710510254794) ) ) {
              result[0] += -0.01744808248108327;
            } else {
              result[0] += 0.06322414938658716;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
              result[0] += -0.019931157263767346;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                result[0] += 0.020057027071791328;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.628996372222901279) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.03896987663174736;
                  } else {
                    result[0] += 0.08439581777713034;
                  }
                } else {
                  result[0] += 0.03283128648444999;
                }
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.0005047868410629614;
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.097527027130127841) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.008227724083941944;
                } else {
                  result[0] += 0.03173468178530916;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                  result[0] += 0.0041955868204191825;
                } else {
                  result[0] += -0.05410497847074001;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.005102721193984358;
                } else {
                  result[0] += 0.037479444260778115;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.966960191726685458) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                      result[0] += -0.0258146071783716;
                    } else {
                      result[0] += -0.005542624295766496;
                    }
                  } else {
                    result[0] += 0.006284250523113186;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.000647270979118814;
                  } else {
                    result[0] += 0.04307593288141467;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.0027974106366656735;
                } else {
                  result[0] += 0.04968128547360702;
                }
              } else {
                result[0] += -0.0043742253411365745;
              }
            } else {
              result[0] += -0.00944117488663518;
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.02182909683363475;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.007046039076843895;
                } else {
                  result[0] += 0.06763385881884419;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.031060218811036933) ) ) {
                    result[0] += -0.004358760300449887;
                  } else {
                    result[0] += 0.04967165346700703;
                  }
                } else {
                  result[0] += -0.03296841849620288;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.97887301445007413) ) ) {
              result[0] += -0.03666467251065816;
            } else {
              result[0] += -0.011744341946819295;
            }
          }
        }
      } else {
        result[0] += 0.0003057114129278324;
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35526132583618342) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                result[0] += 0.001952844702617136;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.049745559692384589) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.00011055316275845049;
                      } else {
                        result[0] += 0.022447387840559458;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.04181545786388558;
                      } else {
                        result[0] += 0.0306344015259577;
                      }
                    }
                  } else {
                    result[0] += 0.0025296107996176085;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.007458396769872956;
                  } else {
                    result[0] += 0.039109438369593276;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.019038060638367996;
              } else {
                result[0] += -0.02827916660486134;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += -0.009695891748426217;
              } else {
                result[0] += 0.0038878809874304783;
              }
            } else {
              result[0] += -0.052937269577890904;
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                result[0] += 0.007892090207031702;
              } else {
                result[0] += -0.0511423862787602;
              }
            } else {
              result[0] += -0.0032265662674130094;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                result[0] += 0.002623734637263534;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.994848489761353427) ) ) {
                  result[0] += -0.014441699170925191;
                } else {
                  result[0] += -0.053981742730670926;
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.007238139582738269;
              } else {
                result[0] += 0.01072620731077566;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
                result[0] += 0.002129343222570946;
              } else {
                result[0] += -0.03564599742796836;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.0820135214513672;
              } else {
                result[0] += -0.014684298723231994;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.6149935722351092) ) ) {
                result[0] += 0.031373386802402224;
              } else {
                result[0] += -0.007357864987259492;
              }
            } else {
              result[0] += -0.017948111397935666;
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.023725908379564124;
          } else {
            result[0] += 0.0019245819379643332;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        result[0] += -0.000388881208692925;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
          result[0] += 0.023776522935190083;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.01114917171215649;
            } else {
              result[0] += 0.08340733346863606;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.008786532456610053;
            } else {
              result[0] += 0.1325624805923611;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
          result[0] += 0.020158698785109022;
        } else {
          result[0] += 0.004276615763017731;
        }
      } else {
        result[0] += -0.027142523403788046;
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
        result[0] += 0.005468001253583955;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.006145371833279162;
        } else {
          result[0] += -0.012136083394536642;
        }
      }
    } else {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.00819838141321236;
          } else {
            result[0] += -0.09662406343588527;
          }
        } else {
          result[0] += -0.0002972055070533603;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
          result[0] += -0.0068433185091088524;
        } else {
          result[0] += -0.02385898612019184;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.00981902321218784;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                result[0] += 0.0026132920438982714;
              } else {
                result[0] += 0.04379464530009469;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.0071441487572281095;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.02849873782251224;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.189540147781372958) ) ) {
                        result[0] += 0.013586702774676995;
                      } else {
                        result[0] += -0.014149478465430618;
                      }
                    }
                  }
                } else {
                  result[0] += -0.03190095247959981;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21668529510498225) ) ) {
                  result[0] += 0.02942308529050578;
                } else {
                  result[0] += -0.02963045979903374;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.610357046127320224) ) ) {
              result[0] += -0.005940716816020892;
            } else {
              result[0] += -0.03595561716005405;
            }
          } else {
            result[0] += -0.03449382482360559;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              result[0] += -0.0039220832453553825;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.047997771251627824;
              } else {
                result[0] += -0.014552681926989667;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.994848489761353427) ) ) {
              result[0] += -0.000877486926728713;
            } else {
              result[0] += 0.006665622191811245;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
            result[0] += 0.0010766769952752983;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.023207635456899578;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                  result[0] += 0.040456429664105203;
                } else {
                  result[0] += 0.01673185940432353;
                }
              } else {
                result[0] += -0.015623565732768997;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.138696432113648349) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.004914226784379627;
          } else {
            result[0] += -0.03733675271425905;
          }
        } else {
          result[0] += 0.022778810935097732;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.196324348449708808) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.014478523004505504;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                  result[0] += 0.056366857473902354;
                } else {
                  result[0] += 0.0028223029443019748;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.966960191726685458) ) ) {
                    result[0] += -0.0044602225910529356;
                  } else {
                    result[0] += -0.027398290063642496;
                  }
                } else {
                  result[0] += -0.04131470598221964;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += -0.0025275240910260046;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.42895507812500178) ) ) {
                    result[0] += 0.02406880795897729;
                  } else {
                    result[0] += -0.028943984599390994;
                  }
                } else {
                  result[0] += 0.040812061464175105;
                }
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06990902777831155;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0051460617302472415;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79285955429077326) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.014632326899693788;
                    } else {
                      result[0] += -0.034184926591564295;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                      result[0] += 0.005354166544024892;
                    } else {
                      result[0] += 0.034726678556260514;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.02136908477341315;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.1414925856985552;
                } else {
                  result[0] += -0.032522111683595346;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.04588476114299199;
              } else {
                result[0] += 0.09763042809571015;
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.087577104568482333) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04741475868161685;
                } else {
                  result[0] += -0.005788519049997159;
                }
              } else {
                result[0] += -0.055045681724901034;
              }
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.008686281220380235;
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.08946606741203317;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.0540185946471215;
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.00991744897534646;
                    } else {
                      result[0] += 0.041659936631129484;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.586156606674195224) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.0029196099900833054;
                } else {
                  result[0] += -0.01377421117306785;
                }
              } else {
                result[0] += -0.02244672232960865;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                result[0] += -0.04955512785264071;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                  result[0] += -0.048067229665116035;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.0015567652412800165;
                  } else {
                    result[0] += -0.06571154212434398;
                  }
                }
              }
            }
          } else {
            result[0] += -0.0004886816835183793;
          }
        } else {
          result[0] += 0.015046355032108641;
        }
      } else {
        result[0] += 0.0006562640373377679;
      }
    } else {
      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
            result[0] += -0.018459488282328594;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.054465182702013906;
            } else {
              result[0] += 0.052505720097795354;
            }
          }
        } else {
          result[0] += 0.005881732251302594;
        }
      } else {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += -0.09318397726968795;
          } else {
            result[0] += -0.010840889384859158;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.00790382737013407;
            } else {
              result[0] += 0.0056062464868032345;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.0044121508770204355;
              } else {
                result[0] += -0.04255568359524427;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.123651981353760654) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.06725514655574856;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.009516513390173575;
                      } else {
                        result[0] += -0.017159124466233362;
                      }
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.008089450194641154;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                          result[0] += -0.015440594421676782;
                        } else {
                          result[0] += 0.0949155894452142;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.03071683057392834;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.030666254530254723;
                } else {
                  result[0] += 0.000471809385155993;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.01992411858632959;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.07644338886499596;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.03926339044719531;
                    } else {
                      result[0] += -0.1462332517806096;
                    }
                  }
                } else {
                  result[0] += -0.03964044046095117;
                }
              } else {
                result[0] += -0.134668672182588;
              }
            }
          } else {
            result[0] += 0.04290151120803745;
          }
        } else {
          result[0] += 0.0003082166099325669;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.01918122460126652;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.02899879173892901;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.0018336364405508458;
                    } else {
                      result[0] += -0.05930547054825567;
                    }
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.008721812484288641;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.97887301445007413) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.109878063201905185) ) ) {
                          result[0] += -0.017325998077930015;
                        } else {
                          result[0] += 0.005932462678471473;
                        }
                      } else {
                        result[0] += 0.017383202981430545;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += 0.0788990673477282;
                  } else {
                    result[0] += 0.015989369141357068;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.09216642598749587;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.028651133119196454;
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.02962836306040355;
                      } else {
                        result[0] += -0.08699052762422249;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.036097586456210376;
                    } else {
                      result[0] += 0.0068352501293859145;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.1183013893670592;
                    } else {
                      result[0] += -0.01906355596066101;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.04577072261613758;
          } else {
            result[0] += -0.007955384444634707;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.007989978842397688;
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.053357196588151946;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += -0.06070456248692618;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                result[0] += -0.035809964744562536;
              } else {
                result[0] += 0.04841805697761938;
              }
            }
          } else {
            result[0] += 0.028344186830396325;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
        result[0] += 0.032190895642689926;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
            result[0] += -0.04890803201563835;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.12364626511417072;
            } else {
              result[0] += -0.03263946712215578;
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.009413904373295853;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                  result[0] += 0.03306730247964184;
                } else {
                  result[0] += -0.015149506204593918;
                }
              } else {
                result[0] += -0.038948419003758464;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0002700010512619901;
              } else {
                result[0] += 0.10346314178962904;
              }
            } else {
              result[0] += -0.03332780004879161;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.016159069146943678;
        } else {
          result[0] += -0.0164116743902186;
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.700598716735840066) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                result[0] += -0.011174923589145113;
              } else {
                result[0] += 0.0155137345333491;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.03863205571736543;
              } else {
                result[0] += -0.010115977815072794;
              }
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.010231761774240346;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.043341875076294833) ) ) {
                          result[0] += -0.030769174528794303;
                        } else {
                          result[0] += 0.0028494106801624356;
                        }
                      } else {
                        result[0] += -0.058226392202236424;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.00424159835068295;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.01463625175318275;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                          result[0] += 0.00479157584941122;
                        } else {
                          if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.328828811645508701) ) ) {
                              result[0] += 0.04194463213954808;
                            } else {
                              result[0] += 0.0119665186728446;
                            }
                          } else {
                            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += 0.050616649590762654;
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += -0.007529730432863995;
                              } else {
                                result[0] += 0.047408245131176974;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                        result[0] += 0.04900480988809178;
                      } else {
                        result[0] += -0.05854528974185912;
                      }
                    } else {
                      result[0] += -0.000312891450214985;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += -0.10941173103025364;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.303973913192749912) ) ) {
                        result[0] += 0.01266900197351231;
                      } else {
                        result[0] += -0.05264349241587434;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.047924273815099144;
              }
            } else {
              result[0] += -0.0021453815688674146;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.016586881135988593;
              } else {
                result[0] += 0.004944931366184236;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.138333082199097124) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.7800335884094256) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.029219565871426108;
                      } else {
                        result[0] += -0.035322407484241;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += -0.005359001220480268;
                        } else {
                          result[0] += 0.07741095113206292;
                        }
                      } else {
                        result[0] += -0.0022883987096368806;
                      }
                    }
                  } else {
                    result[0] += 0.17229497447017086;
                  }
                } else {
                  result[0] += -0.00629418116107665;
                }
              } else {
                result[0] += -0.030416478007000265;
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.04827123344327791;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.017053273158625447;
              } else {
                result[0] += -0.06517672601331509;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
        result[0] += -6.123765021860691e-05;
      } else {
        result[0] += -0.010172888220280979;
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
        result[0] += 0.011055693257798212;
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
              result[0] += 0.03506111691787732;
            } else {
              result[0] += -0.005871467561138093;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
              result[0] += 0.01977675191937254;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.016885292944245293;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.02872761765423746;
                } else {
                  result[0] += -0.07211400676488956;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                result[0] += 0.009108966304017132;
              } else {
                result[0] += -0.01735500245947724;
              }
            } else {
              result[0] += -0.026961384387212774;
            }
          } else {
            result[0] += 0.001878214870960227;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0004839293850812673;
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
            result[0] += -0.011785930472023677;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.018473827993275387;
              } else {
                result[0] += 0.007540291813822375;
              }
            } else {
              result[0] += 0.00984293796640284;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
            result[0] += 0.013021259792676274;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7631258964538592) ) ) {
              result[0] += 0.006328852396179839;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.02514573998350153;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.05573910810359123;
                } else {
                  result[0] += -0.016374052361778604;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
              result[0] += 0.01842749420938986;
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.03490101663822877;
                } else {
                  result[0] += -0.00326530582846536;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.06420914821853804;
                } else {
                  result[0] += -0.02785544838090986;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.011884387393699984;
              } else {
                result[0] += -0.016366261576825192;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.020958508196252584;
                  } else {
                    result[0] += -0.009735380199797533;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.050532319766090066;
                    } else {
                      result[0] += 0.014011603717647354;
                    }
                  } else {
                    result[0] += -0.02444462446349198;
                  }
                }
              } else {
                result[0] += -0.07923283948213666;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.10627247827596045;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                  result[0] += 0.0019140459917308277;
                } else {
                  result[0] += -0.04589818815690712;
                }
              } else {
                result[0] += -0.009071317728617153;
              }
            }
          } else {
            result[0] += 0.023011575183528494;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
        result[0] += -0.008849921137010386;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += 0.017838203067014058;
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                result[0] += 0.033914435055888995;
              } else {
                result[0] += -0.042892288504324805;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                result[0] += 0.014471994913450413;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.008040246603342871;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                      result[0] += 0.012971184661666142;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.04222134331134105;
                      } else {
                        result[0] += 0.022122091053202987;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0095694901256365;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.434232234954834873) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                      result[0] += 0.005930682942551472;
                    } else {
                      result[0] += -0.020424260975636047;
                    }
                  } else {
                    result[0] += 0.05587348337994661;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.027641610062543356;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.01136364414201154;
                      } else {
                        result[0] += -0.041923828739791974;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.013678791875478866;
                          } else {
                            result[0] += -0.0573454553289562;
                          }
                        } else {
                          result[0] += 0.014730041028207179;
                        }
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            result[0] += -0.051142601968400515;
                          } else {
                            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                                  result[0] += 0.008165620532412418;
                                } else {
                                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                                    result[0] += -0.004796881731726403;
                                  } else {
                                    result[0] += -0.03761016504644842;
                                  }
                                }
                              } else {
                                result[0] += -0.0654123499925185;
                              }
                            } else {
                              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.08923208094736691;
                              } else {
                                result[0] += 0.01880242613631871;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                            result[0] += 0.022674995097117512;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11773157119751154) ) ) {
                              result[0] += 0.022226720682566453;
                            } else {
                              result[0] += -0.00239526416859296;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.00604153444813122;
                } else {
                  result[0] += 0.03203711314502338;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.015909424887070884;
                  } else {
                    result[0] += -0.05911610204979001;
                  }
                } else {
                  result[0] += 0.013161047474929539;
                }
              } else {
                result[0] += -0.0010251246811547798;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0005556156717458616;
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
            result[0] += -0.011252243044311307;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
              result[0] += -0.009013081506801706;
            } else {
              result[0] += 0.00774226367922589;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
            result[0] += 0.011886657755921196;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7631258964538592) ) ) {
              result[0] += 0.005318326651136201;
            } else {
              result[0] += -0.014568002139600003;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                result[0] += 0.027773282104164406;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.008153295700881497;
                } else {
                  result[0] += -0.023108382725594984;
                }
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.03871556465692338;
              } else {
                result[0] += -0.01306575393544347;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.025491477648735206;
                } else {
                  result[0] += -0.0013009170382450473;
                }
              } else {
                result[0] += -0.015021115281328073;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.007088573319676584;
                    } else {
                      result[0] += 0.01638254333180118;
                    }
                  } else {
                    result[0] += 0.03337915879933002;
                  }
                } else {
                  result[0] += -0.01582617486715175;
                }
              } else {
                result[0] += -0.07627190025607028;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.09934811444773856;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                  result[0] += 0.0019154288145670478;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.011039172090937587;
                  } else {
                    result[0] += -0.04647764918671183;
                  }
                }
              } else {
                result[0] += -0.008219737599947919;
              }
            }
          } else {
            result[0] += 0.020179607778057218;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.802901029586792436) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.015127961795794238;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.005410785033395951;
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06336843362900947;
                } else {
                  result[0] += -0.017385340966376412;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.04970942039254579;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                    result[0] += -0.0965241339238646;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
                      result[0] += 0.009720621592753408;
                    } else {
                      result[0] += 0.09381700074025218;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.08477887417216773;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
          result[0] += -0.008004918259676098;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.01885533744171129;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.01460840076839807;
                  } else {
                    result[0] += -0.021749248510410053;
                  }
                } else {
                  result[0] += -0.026598463894037368;
                }
              } else {
                result[0] += 0.0327192945709791;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.01635347817798674;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78560066223144709) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.0020164890420878;
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.589327573776246005) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.019827832749232915;
                          } else {
                            result[0] += -0.0005179921267479866;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.06148584572184362;
                            } else {
                              result[0] += 0.020781290522143052;
                            }
                          } else {
                            result[0] += 0.016165265702782306;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.004806595281522197;
                    }
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.970040798187256748) ) ) {
                        result[0] += 0.00012649149456205988;
                      } else {
                        result[0] += 0.04572467622960504;
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.102609157562256748) ) ) {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += -0.024686337073095588;
                            } else {
                              result[0] += 0.021394954250981695;
                            }
                          } else {
                            result[0] += -0.020379062608814554;
                          }
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                              result[0] += -0.14556308473288038;
                            } else {
                              result[0] += -0.03761927701247034;
                            }
                          } else {
                            result[0] += -0.028341955321032515;
                          }
                        }
                      } else {
                        result[0] += 0.07512476748753219;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.01696108713351344;
                      } else {
                        result[0] += -0.010261633795142621;
                      }
                    } else {
                      result[0] += 0.010880417690087825;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.845905780792238104) ) ) {
                      result[0] += 0.007851623052759533;
                    } else {
                      result[0] += 0.06455532704879684;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.208590507507325107) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.196324348449708808) ) ) {
            result[0] += -0.0010710929813744223;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.006961177680410796;
              } else {
                result[0] += -0.04932864638637385;
              }
            } else {
              result[0] += -0.002674034394219572;
            }
          }
        } else {
          result[0] += 0.00034157205414860666;
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.011349904011435744;
          } else {
            result[0] += -0.057689500367119353;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81568002700805842) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                result[0] += -0.007971192637419487;
              } else {
                result[0] += 0.08663221507664608;
              }
            } else {
              result[0] += -0.03765667388382382;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                result[0] += 0.17632010149029126;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.587220668792725498) ) ) {
                  result[0] += -0.10296464027839386;
                } else {
                  result[0] += 0.013584342593990027;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.551017761230469638) ) ) {
                result[0] += 0.04343095589983686;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.954540252685547763) ) ) {
                  result[0] += -0.0769607259505894;
                } else {
                  result[0] += 0.019472521664115075;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.551017761230469638) ) ) {
            result[0] += -0.0025250300939519445;
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.01155620997436146;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
                    result[0] += -0.03937243063373414;
                  } else {
                    result[0] += -0.07541761209595715;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                    result[0] += 0.013568796755380636;
                  } else {
                    result[0] += -0.05165117958450252;
                  }
                }
              } else {
                result[0] += -0.012439654915448371;
              }
            }
          }
        } else {
          result[0] += 0.0034345858396956073;
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.448852539062500444) ) ) {
          result[0] += 0.017889655117343833;
        } else {
          result[0] += -0.03203849925052311;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
          result[0] += 0.00012216629791703888;
        } else {
          result[0] += 0.0337774987104685;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.632926940917970526) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += -0.014394767498401025;
            } else {
              result[0] += -0.06113658117384905;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.0013801987323191066;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                result[0] += -0.05064502499332607;
              } else {
                result[0] += -0.00873842511587782;
              }
            }
          }
        } else {
          result[0] += -0.040916378954201055;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.726826429367066318) ) ) {
          result[0] += 0.002318520581394838;
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += -0.036186912714987224;
            } else {
              result[0] += 0.012051952213850145;
            }
          } else {
            result[0] += -0.029840682504357043;
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.03753351765230817;
          } else {
            result[0] += 0.010407571649854439;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                  result[0] += 0.020269457911891928;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
                    result[0] += -0.001971718263427972;
                  } else {
                    result[0] += -0.060136925603916716;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                      result[0] += 0.01274268881182486;
                    } else {
                      result[0] += -0.03168251095712164;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.02317393068801244;
                    } else {
                      result[0] += -0.023401139414821564;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                    result[0] += 0.01082909029547198;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.08681086818555431;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                            result[0] += -0.010246729469725463;
                          } else {
                            result[0] += 0.03693218039785006;
                          }
                        } else {
                          result[0] += -0.032591330938716544;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.003838300704956943) ) ) {
                        result[0] += -0.0077168714938094865;
                      } else {
                        result[0] += -0.04559261430241812;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.434232234954834873) ) ) {
                result[0] += 0.005334524768387425;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                  result[0] += -0.019702392594471905;
                } else {
                  result[0] += 0.025008614969191045;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.001806651980518925;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.02067964471121909;
                  } else {
                    result[0] += -0.056348143265647015;
                  }
                } else {
                  result[0] += 0.018918632816690352;
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                result[0] += 0.0022319910986089548;
              } else {
                result[0] += -0.04015197679165764;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    result[0] += 0.0003568548493426346;
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.037142778969071616;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.023923481148379028;
                } else {
                  result[0] += 0.05776407407701344;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.07509815582881951;
                } else {
                  result[0] += 0.015097371832643225;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                  result[0] += -0.0017966441958433246;
                } else {
                  result[0] += -0.022472710478597772;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += 0.006753903761126326;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.015660796110303972;
                } else {
                  result[0] += -0.04377169523026566;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                  result[0] += 0.021229059010060343;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.012906155750809384;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.007746719109854014;
                    } else {
                      result[0] += -0.04670983940968353;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.026977669417398098;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.017860956876156815;
                    } else {
                      result[0] += -0.020849862559337868;
                    }
                  }
                } else {
                  result[0] += 0.003306684392421267;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.030591586281667027;
              } else {
                result[0] += -0.004352637912107392;
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += -0.048479777822699334;
                        } else {
                          result[0] += -0.006175687529702652;
                        }
                      } else {
                        result[0] += -0.13939073605996716;
                      }
                    } else {
                      result[0] += -0.0013252681227692987;
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.47860431671142756) ) ) {
                      result[0] += -0.0949779417254506;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.11117400553296651;
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.08522199324108809;
                        } else {
                          result[0] += 0.08081683700941644;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.052321236122223086;
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.689592361450196201) ) ) {
                          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.03591404706639987;
                          } else {
                            result[0] += 0.04656930822993182;
                          }
                        } else {
                          result[0] += 0.058859918070837625;
                        }
                      } else {
                        result[0] += -0.015047290582069082;
                      }
                    } else {
                      result[0] += 0.033902753261118544;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                      result[0] += 0.017053009197009018;
                    } else {
                      result[0] += -0.03926908553796373;
                    }
                  } else {
                    result[0] += -0.03501957776499303;
                  }
                } else {
                  result[0] += 0.0020421649794063004;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.597218394279480425) ) ) {
                  result[0] += -0.01322622456520646;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.0027959939258336303;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                        result[0] += 0.01338683853916179;
                      } else {
                        result[0] += 0.038968661462570685;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                        result[0] += 0.020061458195951343;
                      } else {
                        result[0] += -0.016826212503664017;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0045707626274072944;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                  result[0] += 0.005096166925743958;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.680079460144043857) ) ) {
                    result[0] += -0.008550375602440949;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                      result[0] += -0.04528256087888733;
                    } else {
                      result[0] += -0.02084047249383588;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                      result[0] += 0.006987360023996656;
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.008870079283458494;
                      } else {
                        result[0] += -0.06986016149901264;
                      }
                    }
                  } else {
                    result[0] += -0.00025640035943697313;
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.313553571701050693) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.092434883117676669) ) ) {
                        result[0] += 0.0006795077873522545;
                      } else {
                        result[0] += 0.029340261017325987;
                      }
                    } else {
                      result[0] += -0.0151337632054983;
                    }
                  } else {
                    result[0] += 0.04713053943681862;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.04254252920318248;
        } else {
          result[0] += -0.009434770563098143;
        }
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
          result[0] += -0.026745478976179855;
        } else {
          result[0] += -0.061293134464912025;
        }
      } else {
        result[0] += -0.009258182942171787;
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -9.675875174935042e-05;
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += 0.060701842770731484;
            } else {
              result[0] += -0.001723155895826992;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                result[0] += 0.00378784217975438;
              } else {
                result[0] += -0.09190528146460625;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    result[0] += -0.016005817067124676;
                  } else {
                    result[0] += 0.020856590046385;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.00481656428262614;
                  } else {
                    result[0] += 0.014966318938723753;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.265348911285402167) ) ) {
                    result[0] += -0.03612763657793646;
                  } else {
                    result[0] += 0.021540090857267213;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.901921629905701128) ) ) {
                      result[0] += -2.1324626984189394e-05;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.925687789916993964) ) ) {
                          result[0] += 0.04299973687465339;
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.4822273254394549) ) ) {
                              result[0] += 0.007666675591915456;
                            } else {
                              result[0] += 0.10945447767204382;
                            }
                          } else {
                            result[0] += -0.024419920855070144;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.026011384908639523;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                            result[0] += 0.04395105996951272;
                          } else {
                            result[0] += 0.09824022428397708;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                      result[0] += -0.05317724008740861;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.07666452218280358;
                      } else {
                        result[0] += 0.02483700049588143;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.037208173546140755;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.012933849905679513;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                result[0] += 0.030012551265309707;
              } else {
                result[0] += 0.06914614945831969;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.002473649006729639;
            } else {
              result[0] += 0.05126612495436152;
            }
          } else {
            result[0] += -0.060341859126178767;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
              result[0] += -0.057827480858445524;
            } else {
              result[0] += 0.06664811277509612;
            }
          } else {
            result[0] += -0.08504448430846132;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.010892489879655215;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.566809177398682529) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.722943305969239169) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.07644073194306411;
                  } else {
                    result[0] += 0.027513813088205816;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
                    result[0] += -0.009506223570453423;
                  } else {
                    result[0] += 0.02644892922189066;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.66339445114135831) ) ) {
                  result[0] += -0.032753583313784855;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.008530411535729749;
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.033777048906072214;
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                          result[0] += 0.018424769799399138;
                        } else {
                          result[0] += 0.08621885322443484;
                        }
                      } else {
                        result[0] += -0.00029450384764205924;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += -0.04850779944360811;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.02795108116573641;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.0340725670140673;
                    } else {
                      result[0] += -0.04605062739692503;
                    }
                  } else {
                    result[0] += 0.04519216843401277;
                  }
                }
              }
            }
          } else {
            result[0] += 0.034416377732626166;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
          result[0] += -0.09254053023173892;
        } else {
          result[0] += -0.020567182787371407;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.497191667556763583) ) ) {
            result[0] += -0.006140907252901522;
          } else {
            result[0] += 0.03455383915402486;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
            result[0] += -0.05912169932487086;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)14.32165384292602717) ) ) {
              result[0] += -0.1126737298593499;
            } else {
              result[0] += 0.05305395706226103;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1148419380188006) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.008777987918243918;
            } else {
              result[0] += -0.07185656910672591;
            }
          } else {
            result[0] += 0.03085853088525284;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.624863147735596591) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
              result[0] += 0.012045784402266224;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.029176263852400016;
              } else {
                result[0] += 0.10680827968028453;
              }
            }
          } else {
            result[0] += -0.07253242711798827;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.196324348449708808) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.161602735519410068) ) ) {
          result[0] += 0.0021902614414473025;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.009874764392735947;
            } else {
              result[0] += 0.02826434106917082;
            }
          } else {
            result[0] += -0.03896383544203219;
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.028239112947896147;
            } else {
              result[0] += 0.002167632536111062;
            }
          } else {
            result[0] += -0.059027901949664355;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.011840436062045991;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                  result[0] += 0.0038091918365743397;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += -0.028825451237246343;
                  } else {
                    result[0] += 0.4797147529597643;
                  }
                }
              } else {
                result[0] += -0.04952413731495117;
              }
            } else {
              result[0] += -0.009975768600314206;
            }
          }
        }
      }
    } else {
      result[0] += 0.0074753786468704985;
    }
  } else {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.024900426704528036;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
              result[0] += 0.01482434891197863;
            } else {
              result[0] += -0.005617361326216229;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.338562726974488193) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
                result[0] += -0.00149753157869067;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.0029385715007210155;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                      result[0] += -0.032056055430672706;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                        result[0] += -0.0287496341578483;
                      } else {
                        result[0] += 0.002965967398253517;
                      }
                    }
                  }
                } else {
                  result[0] += -0.04135876368022938;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.0037326054726755683;
              } else {
                result[0] += 0.028933416648697103;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                result[0] += -0.025630387450567806;
              } else {
                result[0] += 0.011286631716238301;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
                  result[0] += 0.0033140957477066028;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.04022819812600478;
                  } else {
                    result[0] += -0.004927776551738563;
                  }
                }
              } else {
                result[0] += -0.013683151210773476;
              }
            }
          } else {
            result[0] += 0.02584367788016297;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
                result[0] += -0.0032537098678231437;
              } else {
                result[0] += 0.019854325018310873;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.0023161313166400422;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.010632822401241375;
                  } else {
                    result[0] += -0.04006130496882648;
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                        result[0] += 0.0351684564005639;
                      } else {
                        result[0] += -0.05046721325301548;
                      }
                    } else {
                      result[0] += 0.0006426878231558918;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
                      result[0] += -0.0033686748711031136;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += 0.039200038743333394;
                      } else {
                        result[0] += -0.027620797261072194;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.005300027494999512;
                } else {
                  result[0] += -0.01921907332745381;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.0027214174874244054;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.799612998962403232) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.04187881637877782;
                        } else {
                          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.016616662425538383;
                          } else {
                            result[0] += -0.08861024153028918;
                          }
                        }
                      } else {
                        result[0] += 0.035438077779996094;
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.022426614398900813;
                      } else {
                        result[0] += 0.026066374933410043;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.230558872222901279) ) ) {
                        result[0] += 0.007622426686086131;
                      } else {
                        result[0] += 0.03687447297333632;
                      }
                    } else {
                      result[0] += 0.0014177028743692385;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.0017338061544929402;
              } else {
                result[0] += 0.02087323972056755;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.0010269194112174317;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.009398768072788624;
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.022058728941503184;
              } else {
                result[0] += -0.05429894297685931;
              }
            }
          } else {
            result[0] += 0.010418898242065319;
          }
        } else {
          result[0] += -0.0020007203798377664;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      result[0] += -0.0003586212786169548;
    } else {
      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.006537807476470501;
      } else {
        result[0] += -0.0010549380492703992;
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.036462449195882624;
              } else {
                result[0] += 0.02606203185875687;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += 0.013461495285179748;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.00816068148745502;
                  } else {
                    result[0] += -0.017330999971606875;
                  }
                } else {
                  result[0] += -0.0246877912333955;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += 0.006374521829656456;
              } else {
                result[0] += -0.02252838459619871;
              }
            } else {
              result[0] += 0.0016595369597380881;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.86655306816101163) ) ) {
                result[0] += 0.018752896712208436;
              } else {
                result[0] += -0.051235420921937516;
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.03866136659294445;
                    } else {
                      result[0] += -0.0015384728739344386;
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += -0.09679323526797613;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.10658769052870216;
                      } else {
                        result[0] += 0.003937236260159356;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.04856409599015432;
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.767332553863526279) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                          result[0] += 0.030111257271578637;
                        } else {
                          result[0] += -0.012121802315790924;
                        }
                      } else {
                        result[0] += -0.044503033413757026;
                      }
                    } else {
                      result[0] += 0.030543454561051727;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.616744756698609287) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                      result[0] += 0.016128590177794874;
                    } else {
                      result[0] += -0.03596848426297947;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
                      result[0] += -0.04243166062869693;
                    } else {
                      result[0] += 0.0017660049711500773;
                    }
                  }
                } else {
                  result[0] += 0.0021209204343612537;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.003509977347632555;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.010910918621756326;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += -0.029493170942142113;
                    } else {
                      result[0] += 0.02676622074339223;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
                        result[0] += -0.008404093668540651;
                      } else {
                        result[0] += 0.012836697540937783;
                      }
                    } else {
                      result[0] += -0.005405805602759787;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.003451026121184847;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.022840957313124535;
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                            result[0] += 0.048302260573999604;
                          } else {
                            result[0] += 0.007405248334364421;
                          }
                        }
                      } else {
                        result[0] += -0.06350985759319998;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += -0.06401988934402515;
                  } else {
                    result[0] += 0.10171835314059818;
                  }
                } else {
                  result[0] += 0.009113987708677468;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += 0.01483794524446224;
                    } else {
                      result[0] += -0.05313889361692717;
                    }
                  } else {
                    result[0] += -0.010359240698790517;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.233438730239869052) ) ) {
                              result[0] += 0.013655174516351993;
                            } else {
                              result[0] += -0.04352502060914026;
                            }
                          } else {
                            result[0] += -0.019849051944687415;
                          }
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.655387401580811435) ) ) {
                              result[0] += -0.03586533489814613;
                            } else {
                              result[0] += -0.0959134580212474;
                            }
                          } else {
                            result[0] += -0.00468900802041556;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.949854612350464755) ) ) {
                          result[0] += 0.007953183247435526;
                        } else {
                          result[0] += 0.08482987443511625;
                        }
                      }
                    } else {
                      result[0] += 0.015211781272747835;
                    }
                  } else {
                    result[0] += -0.04307955055474649;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.03912032186979433;
        } else {
          result[0] += -0.00638394722794786;
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.008488401633803303;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.050987222595374;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.680079460144043857) ) ) {
            result[0] += -0.030796883351858907;
          } else {
            result[0] += 0.01669161761266968;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
        result[0] += 0.0029240991880553004;
      } else {
        result[0] += -0.012287528398000108;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
        result[0] += 0.0018264599682516437;
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.00045495555567426604;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.02126485570872113;
              } else {
                result[0] += -0.11031840864001606;
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.767332553863526279) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += -0.06581124402358236;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += 0.03958119484663755;
                } else {
                  result[0] += -0.02337658785487874;
                }
              }
            } else {
              result[0] += -0.06777239607607252;
            }
          }
        } else {
          result[0] += 0.0003865692949449599;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.587220668792725498) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
        if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.0021457232038041055;
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.006123177782725411;
            } else {
              result[0] += -0.03272573784119556;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                  result[0] += -0.05487486713686915;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.04668442919809711;
                  } else {
                    result[0] += 0.05609975593150701;
                  }
                }
              } else {
                result[0] += 0.00882344787738718;
              }
            } else {
              result[0] += -0.003420849821507621;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                result[0] += 0.020212346919947836;
              } else {
                result[0] += 0.09029963474801778;
              }
            } else {
              result[0] += -0.012461237315686819;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.469231128692627841) ) ) {
                  result[0] += -0.013261757809689127;
                } else {
                  result[0] += -0.04075923627475067;
                }
              } else {
                result[0] += -0.0016180872072988222;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.06930372811225312;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.0823250847980573;
                  } else {
                    result[0] += 0.015947918716022576;
                  }
                } else {
                  result[0] += -0.04346259746316762;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += -0.007580988710097621;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.07989976117546821;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.05524562018427817;
                } else {
                  result[0] += -0.02001715006074016;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.042300975295265136;
                } else {
                  result[0] += -0.010404356902063254;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.027149746506933987;
                  } else {
                    result[0] += 0.0024060242330162106;
                  }
                } else {
                  result[0] += -0.04572974370270267;
                }
              }
            } else {
              if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.009378046005329257;
                } else {
                  result[0] += 0.04810075313069274;
                }
              } else {
                result[0] += 0.009220860131273935;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.176905632019043857) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              result[0] += -0.03911053335362182;
            } else {
              result[0] += -0.003213468147074161;
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.002118382443028334;
            } else {
              result[0] += 0.016134402608039618;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
            result[0] += -0.0018104327106165958;
          } else {
            result[0] += -0.011298289007125551;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.123651981353760654) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                result[0] += -0.02332966734554236;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                  result[0] += 0.010027559057401075;
                } else {
                  result[0] += -0.052470433315170865;
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.012717819344407626;
                } else {
                  result[0] += -0.06573208609862732;
                }
              } else {
                result[0] += -0.05836752350730995;
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.02849901440287359;
                  } else {
                    result[0] += -0.0020485032029928544;
                  }
                } else {
                  result[0] += -0.025347326205281517;
                }
              } else {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.04920738512862888;
                } else {
                  if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.01784489848385631;
                  } else {
                    result[0] += 0.006042748721375529;
                  }
                }
              }
            } else {
              result[0] += -0.028311225379576734;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            result[0] += -0.01662715807608499;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.944137096405030185) ) ) {
              result[0] += 0.015038917181663003;
            } else {
              result[0] += 0.040526949074231704;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += 0.002423967561138815;
    } else {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.007217554781928804;
          } else {
            result[0] += -0.09029105632214282;
          }
        } else {
          result[0] += 0.00017673554606048673;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.568724632263184482) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0020007833256813004;
          } else {
            result[0] += -0.013595718640472676;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
            result[0] += -0.07548485245092057;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += -0.07957707166959427;
            } else {
              result[0] += -0.015862645702248854;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.233438730239869052) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                  result[0] += -0.0021886419692108616;
                } else {
                  result[0] += 0.030630376346896304;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.33441734313965021) ) ) {
                  result[0] += -0.0023019490007313375;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.66339445114135831) ) ) {
                    result[0] += -0.0025190017813185596;
                  } else {
                    result[0] += -0.033765176416416603;
                  }
                }
              }
            } else {
              result[0] += 0.009655011272064454;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.802901029586792436) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += -0.03152022870707679;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.00653966497895714;
                } else {
                  result[0] += -0.030821187787687485;
                }
              }
            } else {
              result[0] += 0.013047628508790519;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                result[0] += -0.009790140018375654;
              } else {
                result[0] += -0.03863197431058721;
              }
            } else {
              result[0] += 0.02013162535426408;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.668367385864259589) ) ) {
                result[0] += 0.001245453667462106;
              } else {
                result[0] += 0.021849646347147835;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.259080410003662998) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                      result[0] += -0.009047053076950497;
                    } else {
                      result[0] += 8.447510985291205e-06;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                        result[0] += 0.006120858722271714;
                      } else {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += -0.014376803219107252;
                                } else {
                                  result[0] += 0.02847452567465519;
                                }
                              } else {
                                result[0] += -0.03801586663276474;
                              }
                            } else {
                              result[0] += -0.04076545427210807;
                            }
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.239300251007080966) ) ) {
                              result[0] += 0.0145909309731637;
                            } else {
                              result[0] += 0.17266644287854963;
                            }
                          }
                        } else {
                          result[0] += 0.011451702986569447;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                          result[0] += 0.010437350724960909;
                        } else {
                          result[0] += 0.03116087617296025;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.17202329635620295) ) ) {
                          result[0] += -0.03743825564052062;
                        } else {
                          result[0] += 0.036531023375226845;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.008143131588817522;
                }
              } else {
                result[0] += 0.06682895663451688;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
          result[0] += -0.0016567866971564913;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.04094598658331638;
          } else {
            result[0] += -0.010092060927698513;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
            result[0] += 0.04161392202937192;
          } else {
            result[0] += 0.0008349647220192865;
          }
        } else {
          result[0] += -0.026189884311606734;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.62696647644043146) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.221818685531617099) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.004261968573900351;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                            result[0] += 0.08752553696993719;
                          } else {
                            result[0] += -0.006923400660683816;
                          }
                        } else {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.01443324738749883;
                          } else {
                            result[0] += -0.05332981524756169;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.008150246198607188;
                    }
                  } else {
                    result[0] += -0.035287688200950594;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                    result[0] += 0.026660176104174062;
                  } else {
                    result[0] += 0.00638257525749177;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                  result[0] += 0.034950078897399615;
                } else {
                  result[0] += -0.0450887719298935;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.26464319229126154) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.353313446044923651) ) ) {
                    result[0] += -0.02467418857214436;
                  } else {
                    result[0] += -0.0019981317095585016;
                  }
                } else {
                  result[0] += -0.055685597143517;
                }
              } else {
                result[0] += 0.013277627383452912;
              }
            }
          } else {
            result[0] += 0.015528818490512611;
          }
        } else {
          result[0] += 0.020556935141508668;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
      result[0] += 0.0012496712062921769;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.0008833050097142884;
          } else {
            result[0] += -0.03125012422505279;
          }
        } else {
          result[0] += -0.031912161684208074;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.009859586911811323;
              } else {
                result[0] += -0.05112719080979677;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                  result[0] += 0.008516139397719795;
                } else {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.004363017071512211;
                  } else {
                    result[0] += 0.10803854274940738;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.019705508422256546;
                  } else {
                    result[0] += -0.06718162643548024;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                    result[0] += 0.02486067056961713;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      result[0] += -0.06232420408313292;
                    } else {
                      result[0] += 0.09543523948694077;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.024388580847537734;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  result[0] += 0.008956212446570022;
                } else {
                  result[0] += -0.021881800479678585;
                }
              }
            } else {
              result[0] += -0.007054291169492532;
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.011142530898461433;
          } else {
            result[0] += -0.06454656369040053;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.233438730239869052) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
            result[0] += 0.001989189883653055;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.003482827687726205;
              } else {
                result[0] += -0.034516670679773966;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.011330190518533073;
              } else {
                result[0] += -0.037324300426469115;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.017299444481259373;
            } else {
              result[0] += 0.019640499930593747;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.668367385864259589) ) ) {
                result[0] += 0.0011384452703935522;
              } else {
                result[0] += 0.02024940144377597;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.259080410003662998) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                      result[0] += -0.008589579503989486;
                    } else {
                      result[0] += -7.874847960917601e-05;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                        result[0] += 0.010006522495663117;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                                result[0] += -0.017487253633827648;
                              } else {
                                result[0] += -0.05472390115780708;
                              }
                            } else {
                              result[0] += 0.001170640709722265;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                              result[0] += 0.01784201685778595;
                            } else {
                              result[0] += -0.01660386902656795;
                            }
                          }
                        } else {
                          result[0] += 0.014190241906417192;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += 0.014336287512563835;
                      } else {
                        result[0] += -0.018923112883409476;
                      }
                    }
                  }
                } else {
                  result[0] += 0.0077487138529285324;
                }
              } else {
                result[0] += 0.06200607794420537;
              }
            }
          }
        }
      } else {
        result[0] += -0.008847063098848494;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.008440201237678751;
            } else {
              result[0] += -0.023729736199214747;
            }
          } else {
            result[0] += 0.007487108418895029;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                result[0] += -0.002888284149371624;
              } else {
                result[0] += -0.04108212241239104;
              }
            } else {
              result[0] += -0.00603543128743921;
            }
          } else {
            result[0] += 0.01777148626909956;
          }
        }
      } else {
        if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
            result[0] += -0.024791451369732723;
          } else {
            result[0] += 0.029930376380687726;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
              result[0] += 0.031045088856115455;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.04674016214373741;
              } else {
                result[0] += -0.002473921742493167;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.395718574523926669) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54296922683715998) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0055870779745577906;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.353313446044923651) ) ) {
                    result[0] += -0.02900094702234803;
                  } else {
                    result[0] += -0.0011199416133148414;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0071020203552178895;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                      result[0] += -0.01219773238143612;
                    } else {
                      result[0] += 0.0299812804115764;
                    }
                  } else {
                    result[0] += 0.040607797379561106;
                  }
                }
              }
            } else {
              result[0] += 0.019381341501334005;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += -0.00040941278523530576;
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.092434883117676669) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.13808324190517654;
                } else {
                  result[0] += 0.0020800528490148087;
                }
              } else {
                result[0] += 0.03475467471059126;
              }
            } else {
              result[0] += -0.0025781976498373367;
            }
          } else {
            result[0] += 0.011572214405613498;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
            result[0] += 0.002795671774179271;
          } else {
            result[0] += -0.007946393937846524;
          }
        }
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
          result[0] += 0.005095755300007388;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.059359570570609116;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                        result[0] += -0.07410226427167281;
                      } else {
                        result[0] += 0.014382738573550583;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                      result[0] += 0.01695538015426409;
                    } else {
                      result[0] += -0.035957040293460044;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    result[0] += -0.0021739447453986704;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
                      result[0] += -0.051594751254783223;
                    } else {
                      result[0] += 0.011830397413982383;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                  result[0] += -0.014507345924152388;
                } else {
                  result[0] += 0.010782894643087536;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.018217148798205227;
                  } else {
                    result[0] += -0.016179785239777412;
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += -0.1548919618840547;
                  } else {
                    result[0] += -0.03409805892708866;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                  result[0] += 0.10003255218139495;
                } else {
                  result[0] += 0.008713435379950084;
                }
              }
            }
          } else {
            result[0] += 0.07124487370846343;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.0005463663156713682;
                    } else {
                      result[0] += 0.02775638898212759;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                        result[0] += 0.007393294115149273;
                      } else {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.05852614172341968;
                        } else {
                          result[0] += -0.016161059534273322;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.400584220886231357) ) ) {
                        result[0] += 0.02276611626966093;
                      } else {
                        result[0] += -0.0033532075453058564;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95751476287841975) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
                          result[0] += 0.03398228420930507;
                        } else {
                          result[0] += -0.020334422613066137;
                        }
                      } else {
                        result[0] += -0.040988599506224674;
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.046629638422097705;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                          result[0] += -0.08711801442840617;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.043341875076294833) ) ) {
                            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.05234231558874611;
                            } else {
                              result[0] += -0.12327639991641481;
                            }
                          } else {
                            result[0] += -0.10239774270628683;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.022322932289091636;
                      } else {
                        result[0] += -0.06270095476179614;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.029014149178962813;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.865389823913576) ) ) {
                          result[0] += -0.021404722302165384;
                        } else {
                          result[0] += 0.01998893408213849;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.00017885772266225215;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.0418797335259534;
                      } else {
                        result[0] += -0.005198511920320595;
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.06615982276957001;
                      } else {
                        result[0] += -0.016985803170445717;
                      }
                    }
                  } else {
                    result[0] += 0.006460030751752233;
                  }
                }
              }
            } else {
              result[0] += -0.02469611069270796;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.48872375488281428) ) ) {
              result[0] += -0.03155338788283573;
            } else {
              result[0] += 0.04675740765106082;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.01559125903774912;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.0003190939022877327;
                } else {
                  result[0] += 0.07772753466517762;
                }
              } else {
                result[0] += -0.05585239555703782;
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.88174462318420499) ) ) {
              result[0] += -0.03427190582241717;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.02239501425620361;
              } else {
                result[0] += 0.023469075929634917;
              }
            }
          }
        }
      } else {
        result[0] += 0.0006714682213825866;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += -0.0005503907674982436;
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.556798219680787021) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
            result[0] += 0.010345965581026705;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
              result[0] += -0.0026214037265770785;
            } else {
              result[0] += 0.011370520765373027;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
            result[0] += 0.003762559114471137;
          } else {
            result[0] += -0.008686349217710018;
          }
        }
      } else {
        result[0] += 0.004175525181165125;
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.11445738636206036;
          } else {
            result[0] += -0.004364816718865628;
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.00396669736860163;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.004647458100787282;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.28202676773071467) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                        result[0] += 0.03379409207105277;
                      } else {
                        result[0] += 0.007630537791880375;
                      }
                    } else {
                      result[0] += 0.0353021938278927;
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                          result[0] += -0.0781095692600973;
                        } else {
                          result[0] += 0.05728036618129165;
                        }
                      } else {
                        result[0] += -0.05076826258431735;
                      }
                    } else {
                      result[0] += 0.027628681339957734;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.010710899472460136;
                  } else {
                    result[0] += -0.016952225861030328;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.845905780792238104) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.007602327830117355;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.021771668021036996;
                  } else {
                    result[0] += -0.06129525181337397;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.27480554580688654) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.001968483270143032;
                      } else {
                        result[0] += -0.041744454354028;
                      }
                    } else {
                      result[0] += 0.012465248760547023;
                    }
                  } else {
                    result[0] += -0.044575028736523074;
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                          result[0] += 0.01794302753004138;
                        } else {
                          result[0] += -0.04775955713622263;
                        }
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.05210090578761091;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70956039428711115) ) ) {
                              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += -0.05274991883924285;
                              } else {
                                result[0] += 0.010637540648295125;
                              }
                            } else {
                              result[0] += -0.05612225810278122;
                            }
                          } else {
                            result[0] += 0.04672664404098624;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.95056009292602717) ) ) {
                        result[0] += 0.023166595906996008;
                      } else {
                        result[0] += 0.14588986444352467;
                      }
                    }
                  } else {
                    result[0] += 0.03118395132760939;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
            result[0] += -0.007360895231968471;
          } else {
            result[0] += -0.031148167773619034;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.003670101025443101;
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.1399618266128145;
                      } else {
                        result[0] += -0.0016494517875157112;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                        result[0] += -0.00881234353634635;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.865389823913576) ) ) {
                          result[0] += -0.0065110282318462795;
                        } else {
                          result[0] += 0.030496670077693418;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += -0.028229162791384495;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
                        result[0] += -0.04972152697673695;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                            result[0] += -0.09509544804088858;
                          } else {
                            result[0] += 0.05224070474135292;
                          }
                        } else {
                          result[0] += 0.07214910549676941;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.0028991557466222103;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                          result[0] += -0.06942915963313233;
                        } else {
                          result[0] += 0.028011693691966502;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.06579434652865566;
                      } else {
                        result[0] += -0.002276705896216599;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                        result[0] += -0.009787097891821405;
                      } else {
                        result[0] += 0.012352233321876514;
                      }
                    } else {
                      result[0] += 0.049644940648815444;
                    }
                  }
                }
              } else {
                result[0] += 0.019973477367893833;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                  result[0] += -0.03506715017167234;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                    result[0] += 0.01224416782423956;
                  } else {
                    result[0] += 0.06806558305167454;
                  }
                }
              } else {
                result[0] += -0.008596879935473924;
              }
            }
          }
        }
      }
    }
  }
}

