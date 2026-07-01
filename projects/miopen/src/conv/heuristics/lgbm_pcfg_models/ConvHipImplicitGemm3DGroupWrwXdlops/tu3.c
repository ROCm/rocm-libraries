
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.010017345491600705;
            } else {
              result[0] += -0.08794789638848544;
            }
          } else {
            result[0] += -0.005506661509051175;
          }
        } else {
          if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.00586547039369457;
          } else {
            result[0] += 0.028363753334468295;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.008766033639873607;
        } else {
          result[0] += 0.05071932728256232;
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -0.002696448603518621;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
          result[0] += -0.04176906713785774;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += -0.06511879667011611;
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.02564212513028253;
            } else {
              result[0] += 0.07491065159975184;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
                result[0] += 0.022164742283977624;
              } else {
                result[0] += -0.04097587825433893;
              }
            } else {
              result[0] += 0.03548653322090203;
            }
          } else {
            result[0] += -0.048406360036350016;
          }
        } else {
          result[0] += -0.08471512295933974;
        }
      } else {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.06475737803875727;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.046585061569255876;
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                  result[0] += -0.09051666146954568;
                } else {
                  result[0] += 0.06203404645257356;
                }
              } else {
                result[0] += 0.024549923624463812;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
              result[0] += 0.024821595215841707;
            } else {
              result[0] += 0.10217569702700709;
            }
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.06915230986959063;
                } else {
                  result[0] += 0.035424107039351366;
                }
              } else {
                result[0] += -0.08795241476444565;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.012291336306524703;
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                      result[0] += 0.11152806825714588;
                    } else {
                      result[0] += 0.01198164650036764;
                    }
                  }
                } else {
                  result[0] += -0.06337495850096604;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += 0.004362180052667988;
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.06838005455628741;
                    } else {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
                              result[0] += 0.18174607126929362;
                            } else {
                              result[0] += -0.0017863152945028283;
                            }
                          } else {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              result[0] += -0.06087778527337373;
                            } else {
                              result[0] += 0.05216004040213669;
                            }
                          }
                        } else {
                          result[0] += -0.017741144976757266;
                        }
                      } else {
                        result[0] += -0.08585482363453703;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                            result[0] += -0.07896680041004847;
                          } else {
                            result[0] += -0.33484465142973396;
                          }
                        } else {
                          result[0] += 0.018034993116486896;
                        }
                      } else {
                        result[0] += -0.08191178660016511;
                      }
                    } else {
                      result[0] += 0.009298082186652628;
                    }
                  } else {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.011749195954225662;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.021331232841953443;
                        } else {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                            result[0] += 0.04946369447874732;
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.0993730609269409;
                            } else {
                              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                                  result[0] += 0.3673676807878834;
                                } else {
                                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                                    result[0] += 0.47001549654071173;
                                  } else {
                                    result[0] += 1.6395834161820027;
                                  }
                                }
                              } else {
                                result[0] += 0.034569526412289005;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += 0.006189588833693645;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += -0.017548119091155615;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.08164982434470403;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.06039419046812007;
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.08795175944898447;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += 0.07394718434681677;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += 0.05209710327044288;
                  } else {
                    result[0] += -0.044461283367266985;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.03463636374064748;
                  } else {
                    result[0] += -0.13574595153945082;
                  }
                } else {
                  result[0] += 0.05132344054394631;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += -0.16694566267304956;
          } else {
            result[0] += -0.006596883571975851;
          }
        } else {
          result[0] += 0.008239844161615516;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.008866656309005483;
        } else {
          result[0] += 0.048884206402111094;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.09395885558853818;
            } else {
              result[0] += -0.026110426626953533;
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += 0.08604999115676659;
              } else {
                result[0] += -0.05549430395122003;
              }
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.03025790586844908;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)23.50000000000000355) ) ) {
                  result[0] += 0.06142126355088993;
                } else {
                  result[0] += -0.03906146545082299;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.03949252756523561;
                    } else {
                      result[0] += 0.0027872599113149448;
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.018178674973369512;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.053756978481790986;
                        } else {
                          result[0] += 0.028223171001229896;
                        }
                      } else {
                        result[0] += -0.044066181664647215;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.020311732923072107;
                      } else {
                        result[0] += -0.06254708463454774;
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.024971062096252866;
                        } else {
                          result[0] += 0.028521809492221162;
                        }
                      } else {
                        result[0] += 0.17673527016410517;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                      result[0] += 0.006834617569615534;
                    } else {
                      result[0] += -0.03593760674159026;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += 0.08956235560199031;
                } else {
                  result[0] += -0.025269218967222475;
                }
              }
            } else {
              result[0] += -0.03544362042821302;
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    result[0] += -0.08541858820269166;
                  } else {
                    result[0] += 0.05023284085508223;
                  }
                } else {
                  result[0] += 0.015886775616579814;
                }
              } else {
                result[0] += 0.02179640693643863;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.00209464443355755;
              } else {
                result[0] += -0.03549777296977571;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.22043503548645116;
          } else {
            result[0] += -0.007833967799961879;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
              result[0] += -0.01187887386234999;
            } else {
              result[0] += -0.06871589427788578;
            }
          } else {
            result[0] += 0.0006548618709685171;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
        result[0] += 0.021972042634678723;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0028412562166727027;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.06566227804868749;
                    } else {
                      result[0] += -0.06704284071792252;
                    }
                  } else {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.010493555712190352;
                    } else {
                      result[0] += 0.07645204563441887;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                    result[0] += 0.030531452778918684;
                  } else {
                    result[0] += -0.03190755851103352;
                  }
                } else {
                  result[0] += 0.015342252413571811;
                }
              }
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                result[0] += -0.019146217321954713;
              } else {
                result[0] += -0.08330986398270548;
              }
            }
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                result[0] += -0.022880085189397933;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.06815012561619874;
                } else {
                  result[0] += 0.12673779725110826;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                result[0] += 0.034546363796379445;
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += 0.00461391316348861;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.09298088529065152;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.08670260707336305;
                    } else {
                      result[0] += 0.04782299052031689;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += 0.004793442985802219;
            } else {
              result[0] += 0.04250384398887285;
            }
          } else {
            result[0] += 0.0017694642914254715;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += -0.018101822452758185;
      } else {
        result[0] += 0.03188906448475664;
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.0002062908711968096;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                result[0] += 0.016037824790844715;
              } else {
                result[0] += -0.09506514679191296;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += 0.12240211421059435;
                } else {
                  result[0] += 0.03243722877231333;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
                      result[0] += -0.07674260092530472;
                    } else {
                      result[0] += 0.1326870513191555;
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                        result[0] += -0.11126642303480645;
                      } else {
                        result[0] += 0.13175546418114456;
                      }
                    } else {
                      result[0] += 0.1046765114784686;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.058868422685560466;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                      result[0] += -0.055545882283811236;
                    } else {
                      result[0] += 0.03359882430577117;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.05262160016739609;
        }
      } else {
        result[0] += 0.07100836993214496;
      }
    } else {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
            result[0] += 0.17153579379539638;
          } else {
            result[0] += -0.008223697178841687;
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              result[0] += -0.03641806170921145;
            } else {
              result[0] += -0.08352239910317555;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.008034116601002598;
            } else {
              result[0] += -0.040191775242366355;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                result[0] += -0.016473666060014724;
              } else {
                result[0] += 0.054335597237864464;
              }
            } else {
              result[0] += 0.06445302123032251;
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.285887241363526279) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += -0.09339118217610598;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.6749763488769549) ) ) {
                  result[0] += 0.006761302346029598;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.1081557465854052;
                  } else {
                    result[0] += -0.0031572118298411967;
                  }
                }
              }
            } else {
              result[0] += -0.07167931077353419;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            result[0] += 0.003993737303081352;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += 0.05131654298724654;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.95906782150268732) ) ) {
                        result[0] += -0.09048559290640264;
                      } else {
                        result[0] += -0.01679361585199455;
                      }
                    }
                  } else {
                    result[0] += -0.1030834328171762;
                  }
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.023174981479381157;
                  } else {
                    result[0] += 0.00192427964436576;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.01805678758179461;
                  } else {
                    result[0] += -0.07345951556367339;
                  }
                } else {
                  result[0] += 0.03988742360327987;
                }
              }
            } else {
              result[0] += -0.060425183741501534;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.010009182697237138;
      } else {
        result[0] += -0.08323557675439613;
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.05019412542739013;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
          result[0] += -0.07228370393781254;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.487163543701172763) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
                    result[0] += 0.034441356725433823;
                  } else {
                    result[0] += -0.048961649865540746;
                  }
                } else {
                  result[0] += -0.06460466121501256;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.02900028228759943) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
                    result[0] += -0.09328205690695868;
                  } else {
                    result[0] += -0.002994963287273126;
                  }
                } else {
                  result[0] += 0.06713952138992506;
                }
              }
            } else {
              result[0] += -0.04587292991506417;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.08972923195989713;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.743881702423096591) ) ) {
                  result[0] += -0.028331975973741094;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.1907894721655537;
                  } else {
                    result[0] += -0.05404691386406005;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.07162030012405197;
                  } else {
                    result[0] += -0.007922108103167019;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.013084210030547275;
                  } else {
                    result[0] += -0.06621012887501118;
                  }
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.01781688623624636;
                  } else {
                    result[0] += -0.010953057631139351;
                  }
                } else {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.032145735416931324;
                  } else {
                    result[0] += -0.06985162263356962;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += 0.0026655005457785195;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.05565418698343763;
            } else {
              result[0] += 0.1267817056649768;
            }
          }
        } else {
          result[0] += -0.06562677952464978;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += -0.03196640339381989;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.008995109970481774;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                result[0] += -0.05000376454548336;
              } else {
                result[0] += 0.03642522713927068;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
            result[0] += -0.13955981555847885;
          } else {
            result[0] += -0.0064834016694910014;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += -0.00262909722678818;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
            result[0] += -0.03900315935504709;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
              result[0] += -0.05662748282247391;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.07266392101982187;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.07895320749811202;
                    } else {
                      result[0] += 0.059674336139885875;
                    }
                  } else {
                    result[0] += 0.032359663910727775;
                  }
                } else {
                  result[0] += 0.08740440075243538;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.65906000137329146) ) ) {
          result[0] += -0.055601410147436595;
        } else {
          result[0] += 0.48460644489414206;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
            result[0] += 0.001343616891949967;
          } else {
            result[0] += -0.060590295434123954;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.0323534428769274;
            } else {
              result[0] += 0.058015527125653714;
            }
          } else {
            result[0] += -0.03344763273737184;
          }
        }
      } else {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
            result[0] += 0.06649591111941631;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.016375637830477196;
                } else {
                  result[0] += -0.1299096925580126;
                }
              } else {
                if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0021976344777582158;
                } else {
                  result[0] += 0.0647813731191368;
                }
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.08684420577100521;
              } else {
                result[0] += 0.0170460058593991;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.03567936607558255;
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01328222607054663;
              } else {
                result[0] += -0.08410606079505789;
              }
            } else {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += 0.020285457103903906;
                  } else {
                    result[0] += -0.05274385118335653;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.02900028228759943) ) ) {
                    result[0] += -0.044467437436644434;
                  } else {
                    result[0] += 0.06439289594013993;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.054825730891788854;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                          result[0] += -0.03307960752736475;
                        } else {
                          result[0] += 0.08470119251612515;
                        }
                      } else {
                        result[0] += 0.0026724791252085954;
                      }
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                        result[0] += -0.01884727788518751;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.048069068583248664;
                        } else {
                          result[0] += 0.0658957283507021;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                              if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                                  result[0] += 0.0262649532176863;
                                } else {
                                  result[0] += -0.033932965134246776;
                                }
                              } else {
                                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                                    result[0] += -0.20989040332483047;
                                  } else {
                                    result[0] += 0.018445901323824895;
                                  }
                                } else {
                                  result[0] += -0.03205614628899161;
                                }
                              }
                            } else {
                              result[0] += -0.0963510187767358;
                            }
                          } else {
                            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.18360908157040748;
                            } else {
                              result[0] += -0.08030011212491857;
                            }
                          }
                        } else {
                          result[0] += -0.0011335401405551623;
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += -0.005730063463430569;
                        } else {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                            result[0] += -0.014171869026445209;
                          } else {
                            result[0] += 0.040956435812294724;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.009608083560010071;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.0010827197428751089;
        } else {
          result[0] += -0.07536609287098921;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
          result[0] += 0.03405582593151284;
        } else {
          result[0] += -0.023107662923565628;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += 0.024162784393644728;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.00401885032029145;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
              result[0] += -0.010073792158503188;
            } else {
              result[0] += 0.041609375303886136;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.05575571381111778;
        } else {
          result[0] += 0.01305510081927505;
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -0.0025770740306986385;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
          result[0] += -0.04189874511687741;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.06089239071680163;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.06774073800833709;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                result[0] += 0.003843484262321812;
              } else {
                result[0] += 0.04922290560173408;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
          result[0] += 0.003506598314104389;
        } else {
          result[0] += -0.07832173833929378;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.04397755657477975;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.10174805539829504;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += 0.03400436407686055;
                  } else {
                    result[0] += -0.03352059125667292;
                  }
                } else {
                  result[0] += -0.07088683904847763;
                }
              } else {
                result[0] += -0.05297193152640417;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.816582441329956943) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.10256392018104882;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += -0.08968946859310974;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += -0.07252219583254331;
                        } else {
                          result[0] += 0.011802799223578554;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.05832576283172291;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.347890853881836826) ) ) {
                      result[0] += -0.04647417245742984;
                    } else {
                      result[0] += 0.030825070607912432;
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.20821138348678403;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.1479520420880491;
                      } else {
                        result[0] += 0.07942926619998991;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.01781839053511949;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.03760938612680373;
                } else {
                  result[0] += 0.0812173900875276;
                }
              } else {
                result[0] += -0.11433195286109657;
              }
            } else {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 7.858722939073451e-05;
              } else {
                result[0] += 0.050712974257878456;
              }
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.03430340048851536;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                    result[0] += 0.09705160341796386;
                  } else {
                    result[0] += 0.04096691445703603;
                  }
                }
              } else {
                result[0] += -0.014066383002789846;
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                  result[0] += -0.09057631543958224;
                } else {
                  result[0] += -0.02611505789928531;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.019517061792165717;
                  } else {
                    result[0] += -0.010970442113756973;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.11214395329420684;
                      } else {
                        result[0] += -0.021673795346525124;
                      }
                    } else {
                      result[0] += -0.07595698924393487;
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
                          result[0] += 0.2593415994898545;
                        } else {
                          result[0] += 0.07739569662959718;
                        }
                      } else {
                        result[0] += -0.03734278549077344;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.012897195184549943;
                        } else {
                          result[0] += -0.07815225870422048;
                        }
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.02078935775979624;
                        } else {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.16341957577580155;
                          } else {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.07140097770327977;
                            } else {
                              result[0] += 1.570762459562899;
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
        } else {
          result[0] += -0.07149401440891962;
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.04775526942750413;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
            result[0] += 0.18581210064622794;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                result[0] += 0.03718625928261957;
              } else {
                result[0] += -0.026751970829241342;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                result[0] += -0.033579491106867906;
              } else {
                result[0] += 0.17795753449587087;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.2376437792436181;
            } else {
              result[0] += 0.0065614738699459;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.05654792261182835;
            } else {
              result[0] += -0.08076444267741384;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += 0.0024285818039189028;
          } else {
            result[0] += 0.01755996124892408;
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.008147411563820836;
          } else {
            result[0] += 0.09976132439465385;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.045583644678897896;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.04585028520771222;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.04984070190086058;
              } else {
                result[0] += 0.009337798387101902;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += -0.002602805408571291;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
              result[0] += 0.011123311778144653;
            } else {
              result[0] += -0.09026625810478067;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
              result[0] += 0.042332623854810694;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                  result[0] += -0.0062529164923953925;
                } else {
                  result[0] += -0.09649011932111282;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.05225821321116823;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.06239704361780336;
                  } else {
                    result[0] += -0.16975604175900255;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.04828039345274026;
      }
    }
  } else {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.851041555404663974) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
              result[0] += 0.0743369692032727;
            } else {
              result[0] += -0.02324251562218595;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.233438730239869052) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                result[0] += 0.03323563825024698;
              } else {
                result[0] += -0.028494289570731236;
              }
            } else {
              result[0] += -0.010653684685608802;
            }
          }
        } else {
          result[0] += -0.037281473810175374;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
          result[0] += 0.03130009045802686;
        } else {
          result[0] += -0.05057815986304795;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
          result[0] += -0.06790959805460382;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
              result[0] += 0.021662198797424058;
            } else {
              result[0] += 0.22061206695220537;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.11244866514092143;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.05168128509888756;
              } else {
                result[0] += 0.09160122012418569;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.06210554798025747;
                } else {
                  result[0] += 0.09108621934845851;
                }
              } else {
                result[0] += -0.028201224517160086;
              }
            } else {
              result[0] += 0.011750833281886116;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.063810348510743076) ) ) {
                  result[0] += -0.07512515303940148;
                } else {
                  result[0] += 0.0054130399287397254;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.02793806516444948;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                      result[0] += 0.06749758286400691;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.12351747631564323;
                          } else {
                            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                                result[0] += -0.07194026490647228;
                              } else {
                                result[0] += -0.0036935996286791986;
                              }
                            } else {
                              result[0] += -0.09066052406890827;
                            }
                          }
                        } else {
                          result[0] += -0.00675852383162907;
                        }
                      } else {
                        result[0] += 0.0028621104258394186;
                      }
                    }
                  }
                } else {
                  result[0] += -0.07853102598966998;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.055710702933891014;
                } else {
                  result[0] += 0.0010276519541327406;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.025976696066981372;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.04823215580664958;
                    } else {
                      result[0] += 0.0899357500881477;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += 0.02399432155289895;
                  } else {
                    result[0] += 0.06539584182224394;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.06638703587401855;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.017070776466923242;
              } else {
                result[0] += 0.04755529077903452;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.012300496994297264;
              } else {
                result[0] += -0.03862883190918953;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.003365213812271001;
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0010345093236491627;
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.009957272985954266;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                result[0] += -0.03358943305545375;
              } else {
                result[0] += -0.1015261018592668;
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.12136047176338116;
                } else {
                  result[0] += -0.002455503971900577;
                }
              } else {
                result[0] += -0.008527180198677152;
              }
            } else {
              result[0] += -0.024363391635496827;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
            result[0] += 0.0012253770844785948;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.06964441351242139;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
                  result[0] += -0.05033690703061108;
                } else {
                  result[0] += -0.005548614696828196;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.05169331702780092;
              } else {
                result[0] += -0.011654670068238757;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.08208316881340111;
              } else {
                result[0] += 0.02653027401043184;
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.06588510008281856;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.00038985723369933995;
                } else {
                  result[0] += -0.05054780167721547;
                }
              }
            }
          } else {
            result[0] += 0.008921066002429834;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
          result[0] += 0.0013197985824045901;
        } else {
          result[0] += -0.07821919646954639;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.04359405500714412;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
            result[0] += 0.11837343664509202;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                    result[0] += 0.03139494146084417;
                  } else {
                    result[0] += -0.032167918030092314;
                  }
                } else {
                  result[0] += -0.05837158325235442;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.816582441329956943) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.09879704496518277;
                      } else {
                        result[0] += -0.039970883021014786;
                      }
                    } else {
                      result[0] += 0.055918701510069016;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.009141406520206258;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.13167101297753692;
                        } else {
                          result[0] += 0.061776320883981266;
                        }
                      } else {
                        result[0] += 0.17969646406711323;
                      }
                    }
                  }
                } else {
                  result[0] += 0.017302797138773172;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.07637114614105926;
              } else {
                result[0] += -0.08116500815800869;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
          result[0] += -0.009225518009143929;
        } else {
          result[0] += -0.07539206025115641;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.026644219496570754;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += 0.10088497280140965;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                      result[0] += 0.41600800297187945;
                    } else {
                      result[0] += -0.040358021012784684;
                    }
                  }
                }
              } else {
                result[0] += -0.012861983415829349;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                  result[0] += 0.0043009053356292066;
                } else {
                  result[0] += -0.04605416832792099;
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                    result[0] += 0.002970624024994253;
                  } else {
                    result[0] += 0.19243669140478328;
                  }
                } else {
                  result[0] += -0.005191689752708537;
                }
              }
            }
          } else {
            result[0] += -0.06433475075561337;
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.0701598914767017;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.014654307005975312;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                      result[0] += 0.37225445631166787;
                    } else {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                        result[0] += 0.2068986316983979;
                      } else {
                        result[0] += -0.04717594612917096;
                      }
                    }
                  }
                } else {
                  result[0] += -0.09045705843400759;
                }
              } else {
                result[0] += -0.06759488506598603;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += 0.08409284801915487;
              } else {
                result[0] += -0.07718666350824037;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += -0.018889606208883834;
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05903331544702622;
                } else {
                  result[0] += 0.16487450783547997;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.001192568044657876;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.09290101033171139;
                } else {
                  result[0] += 0.015100397227454541;
                }
              }
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)11.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += -0.01181670404909314;
                } else {
                  result[0] += 0.022751818799690605;
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                  result[0] += -0.0544131912754686;
                } else {
                  result[0] += 0.0049295735588872295;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.0013674488997689447;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += 0.04157988401081242;
                  } else {
                    result[0] += -0.04261110653137632;
                  }
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                    result[0] += 0.015507797523045869;
                  } else {
                    result[0] += -0.08231667186332758;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                  result[0] += 0.03326247289794001;
                } else {
                  result[0] += -0.06404782835740923;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
            result[0] += -0.0018697435277756796;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.05405911607180716;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.054434566168290226;
                } else {
                  result[0] += -0.005059727863869332;
                }
              }
            } else {
              result[0] += 0.002164845173967575;
            }
          }
        }
      } else {
        result[0] += 0.00014441368281721808;
      }
    } else {
      result[0] += -0.010258913225514988;
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      result[0] += 0.019720534792987587;
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          result[0] += -0.039963887519620475;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
            result[0] += -0.05312049487226936;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
              result[0] += 0.10540680002128687;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.025187934238007283;
                  } else {
                    result[0] += -0.04954804400725647;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.04058665847871924;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.050140449199844686;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                            result[0] += -0.08917317318110468;
                          } else {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                              result[0] += -0.06648609174400792;
                            } else {
                              result[0] += 0.0042978787063544625;
                            }
                          }
                        } else {
                          result[0] += 0.01905059238732862;
                        }
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.025194474013828946;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.816582441329956943) ) ) {
                            result[0] += -0.006158299203525015;
                          } else {
                            result[0] += -0.17857486665087546;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                      result[0] += 0.0601297172794428;
                    } else {
                      result[0] += -0.03947383074689778;
                    }
                  } else {
                    result[0] += 0.012150766104705458;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.03700446903349542;
                    } else {
                      result[0] += -0.021182221964447835;
                    }
                  } else {
                    result[0] += -0.08940405362758636;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
            result[0] += 0.04812830524484184;
          } else {
            result[0] += -0.05984725767520195;
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.04243183420254787;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                      result[0] += -0.06636176153790964;
                    } else {
                      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.03901188983368294;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                            result[0] += -0.21028466615527758;
                          } else {
                            result[0] += 0.018639770677240746;
                          }
                        } else {
                          result[0] += -0.01927595213210609;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
                        result[0] += 0.1094440863310239;
                      } else {
                        result[0] += 0.024427851668738246;
                      }
                    } else {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                        result[0] += -0.06561980089557913;
                      } else {
                        result[0] += 0.021481365910012838;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.03121649043566474;
              }
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                    result[0] += -0.08316162020960524;
                  } else {
                    result[0] += 0.004485080926347332;
                  }
                } else {
                  result[0] += -0.0629210431506991;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.012121820669895242;
                } else {
                  result[0] += 0.04745285623432782;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                result[0] += 0.030095969477855256;
              } else {
                result[0] += -0.06179077098134044;
              }
            } else {
              result[0] += -0.03511164918895754;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    result[0] += -0.00190467904114713;
  } else {
    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.016866709802315354;
                      } else {
                        result[0] += -0.11551606533096398;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                            result[0] += 0.026747141291378014;
                          } else {
                            result[0] += -0.04112610444003889;
                          }
                        } else {
                          result[0] += 0.06467786842157303;
                        }
                      } else {
                        result[0] += -0.06856046827779404;
                      }
                    }
                  } else {
                    result[0] += 0.04761214009506488;
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.04400765141632139;
                      } else {
                        result[0] += -0.03613383131247782;
                      }
                    } else {
                      result[0] += -0.049569547015386704;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            result[0] += 0.03512312767800199;
                          } else {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                                  result[0] += -0.05878064709434516;
                                } else {
                                  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                                    result[0] += 0.031080407317705186;
                                  } else {
                                    result[0] += -0.031021955068402662;
                                  }
                                }
                              } else {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
                                  result[0] += 0.07578949976571307;
                                } else {
                                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                    result[0] += 0.03110757680887294;
                                  } else {
                                    result[0] += -0.01166275423129521;
                                  }
                                }
                              }
                            } else {
                              result[0] += 0.05184543593216448;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                            result[0] += -0.07249894945103684;
                          } else {
                            result[0] += 0.0481175078069978;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30829524993896662) ) ) {
                              result[0] += 0.05054349534241717;
                            } else {
                              result[0] += -0.07819604983226344;
                            }
                          } else {
                            result[0] += -0.02093488810720314;
                          }
                        } else {
                          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                            result[0] += -0.08881898872284605;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                              result[0] += 0.08438805823452239;
                            } else {
                              result[0] += 0.01621366380826386;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                            result[0] += 0.010433244622383448;
                          } else {
                            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.07804228994000562;
                            } else {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                                  result[0] += 0.07306923407158393;
                                } else {
                                  result[0] += -0.09391299669003807;
                                }
                              } else {
                                result[0] += -0.06965379362678849;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.10498711548936757;
                        }
                      } else {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
                            result[0] += 0.03217865376008709;
                          } else {
                            result[0] += -0.08635981292672051;
                          }
                        } else {
                          result[0] += -0.06864070011750295;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.06741548870815887;
                } else {
                  result[0] += -0.00481548647433238;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                result[0] += -0.023732425933355995;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.07641011171875904;
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01339373249148898;
                    } else {
                      result[0] += -0.14474719793115268;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                      result[0] += -0.027676974328207857;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += -0.08854242765739728;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.04823066121498236;
                          } else {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.07988303885804686;
                            } else {
                              result[0] += 0.024269516802598604;
                            }
                          }
                        } else {
                          result[0] += 0.06069434551123801;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.005323220987421648;
            } else {
              result[0] += -0.06223898427083763;
            }
          }
        } else {
          result[0] += -0.05442662958009599;
        }
      } else {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.051490868740001965;
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.06810967382715626;
          } else {
            result[0] += 0.04165288533744778;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0035887230737709948;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.0437893832927941;
                } else {
                  result[0] += -0.1179771633184525;
                }
              } else {
                result[0] += -0.01712987913915222;
              }
            }
          } else {
            result[0] += 0.02626564546647915;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
              result[0] += -0.012800242718234493;
            } else {
              result[0] += 0.06775826626549704;
            }
          } else {
            result[0] += -0.00638812285584802;
          }
        }
      } else {
        result[0] += 0.021579694567873644;
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    result[0] += -0.0019289223339805266;
  } else {
    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.10413977256583491;
            } else {
              result[0] += -0.058003168628959656;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.030219678939373773;
              } else {
                result[0] += 0.12183458235464989;
              }
            } else {
              result[0] += -0.043271034190767936;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
            result[0] += 0.065840958795192;
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += 0.062401497559724586;
                    } else {
                      result[0] += 0.024049199056394903;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                        result[0] += 0.022910318595123644;
                      } else {
                        result[0] += -0.07580394492381863;
                      }
                    } else {
                      result[0] += -0.04185517444562742;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.0675837425572501;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                        result[0] += -0.05688265610456537;
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                          result[0] += 0.15184084538491005;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.12453930147300861;
                          } else {
                            result[0] += 0.06774214334698583;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                            result[0] += 0.029284626370955094;
                          } else {
                            result[0] += -0.026047637822704434;
                          }
                        } else {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                              result[0] += 0.059991671590996866;
                            } else {
                              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                                  result[0] += 0.08666898343493884;
                                } else {
                                  result[0] += -0.0563893748571429;
                                }
                              } else {
                                result[0] += 0.04052015828702628;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                              result[0] += -0.10524778997767732;
                            } else {
                              result[0] += 0.12228622201658966;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.08477114603964843;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.01393975146349864;
                            } else {
                              result[0] += -0.07454203580645755;
                            }
                          } else {
                            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += 0.07272476273363636;
                            } else {
                              if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                                result[0] += -0.09285035090480132;
                              } else {
                                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                                  result[0] += 0.0151329841998294;
                                } else {
                                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
                                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                                        result[0] += 0.09812584302186636;
                                      } else {
                                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                                            result[0] += 0.07872508031638839;
                                          } else {
                                            result[0] += -0.07626539939922872;
                                          }
                                        } else {
                                          result[0] += -0.03438085512539448;
                                        }
                                      }
                                    } else {
                                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                          result[0] += 0.03488455925319;
                                        } else {
                                          result[0] += -0.028977933569223215;
                                        }
                                      } else {
                                        result[0] += 0.04882593038590188;
                                      }
                                    }
                                  } else {
                                    result[0] += -0.0619630773901927;
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.816582441329956943) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                            result[0] += -0.013600163519519662;
                          } else {
                            result[0] += -0.1563608237333006;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                            result[0] += -0.3257377293620928;
                          } else {
                            result[0] += -0.06650183893773436;
                          }
                        }
                      } else {
                        result[0] += 0.020967156920812777;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                  result[0] += -0.054822949654394604;
                } else {
                  result[0] += 0.012888558464555971;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                result[0] += 0.007130270253678699;
              } else {
                result[0] += -0.07151200941085864;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
            result[0] += -0.0031142893614284836;
          } else {
            result[0] += 0.04022650534461633;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.03636148562825405;
                    } else {
                      result[0] += 0.020291024342062067;
                    }
                  } else {
                    result[0] += -0.08422063198808886;
                  }
                } else {
                  result[0] += -0.0825973711827751;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                    result[0] += -0.006801713536655692;
                  } else {
                    result[0] += 0.08998346756358266;
                  }
                } else {
                  result[0] += -0.0889917715622049;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03440881666722716;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04948935957031913;
                } else {
                  result[0] += 0.03590498085668025;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
              result[0] += -0.007527125649495106;
            } else {
              result[0] += 0.1679910675732823;
            }
          }
        }
      }
    } else {
      result[0] += -0.11210490301671899;
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.00033284083847143203;
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                  result[0] += 0.012865375741551225;
                } else {
                  result[0] += -0.06388387276406936;
                }
              } else {
                result[0] += -0.04195250620280909;
              }
            } else {
              result[0] += -0.113054307216267;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.035150566583233796;
                } else {
                  result[0] += 0.030565523171538145;
                }
              } else {
                result[0] += -0.036826385146677264;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                result[0] += -0.10010355606402979;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.04455245257841785;
                  } else {
                    result[0] += 0.015778620573542925;
                  }
                } else {
                  result[0] += -0.06930295972375994;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.148096539672446;
                  } else {
                    result[0] += -0.017147745841708174;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.215905904769898349) ) ) {
                    result[0] += -0.08036838814555211;
                  } else {
                    result[0] += 0.01711502804394914;
                  }
                }
              } else {
                result[0] += -0.004907512654317434;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += -0.04052152546467393;
                } else {
                  result[0] += 0.008212475518498588;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.046601166706600604;
                  } else {
                    result[0] += 0.03695733290577167;
                  }
                } else {
                  result[0] += -0.0724784540196072;
                }
              }
            }
          }
        }
      } else {
        result[0] += 8.690044221886932e-05;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.06280346052756992;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.06613431425653646;
                    } else {
                      result[0] += -0.11230693987797127;
                    }
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += 0.09287208769728161;
                        } else {
                          result[0] += -0.03713914908364014;
                        }
                      } else {
                        result[0] += -0.11046468984101009;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                        result[0] += -0.1611354493526755;
                      } else {
                        result[0] += -0.03753630629276563;
                      }
                    }
                  }
                } else {
                  result[0] += 0.017666912111222576;
                }
              } else {
                result[0] += 0.06813781873189582;
              }
            } else {
              result[0] += -0.06567297968208193;
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.0874703590832247;
          } else {
            result[0] += -0.03531814147001854;
          }
        }
      } else {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += -0.016842252439643483;
        } else {
          result[0] += 0.01689772788695788;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.009268917159765128;
      } else {
        result[0] += 0.024741660660171732;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
            result[0] += 0.013342898090914775;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.02900028228759943) ) ) {
              result[0] += -0.04571960277486922;
            } else {
              result[0] += 0.0642928265342951;
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
              result[0] += 0.035124698309358125;
            } else {
              result[0] += -0.06004696773278248;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.09171914605289963;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.009822586706025408;
                      } else {
                        result[0] += -0.12449511030505289;
                      }
                    } else {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += 0.09039887217077708;
                      } else {
                        result[0] += -0.006348958893151817;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                        result[0] += -0.06214795414983407;
                      } else {
                        result[0] += -0.00430542203663708;
                      }
                    } else {
                      result[0] += -0.0038783952140098916;
                    }
                  }
                } else {
                  result[0] += -0.08429561605310634;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.054448514463989905;
                } else {
                  result[0] += 0.0012383289662935126;
                }
              } else {
                result[0] += 0.017310130671526865;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
          result[0] += 0.060298824806121476;
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += 0.0005678676667261015;
              } else {
                result[0] += -0.061762116121158284;
              }
            } else {
              result[0] += 0.01481444014198028;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
              result[0] += -0.06663625882047772;
            } else {
              result[0] += 0.03867172601805907;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)10.00000000000000178) ) ) {
      if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.012659371150621067;
          } else {
            result[0] += 0.0038018384388823797;
          }
        } else {
          result[0] += 0.017938298338503897;
        }
      } else {
        result[0] += -0.0024813686624237676;
      }
    } else {
      result[0] += -0.09576815896563148;
    }
  } else {
    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += 0.019007532403460023;
                    } else {
                      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += 0.009867249692108122;
                        } else {
                          result[0] += -0.034722012815487806;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
                          result[0] += 0.09601766992751644;
                        } else {
                          result[0] += -0.08981422101485217;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.02691114029795662;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                          if ( UNLIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.023130091342917755;
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                                result[0] += 0.044170186545014804;
                              } else {
                                result[0] += -0.10272096897731027;
                              }
                            } else {
                              result[0] += 0.1482636132349013;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.031000674656720757;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                              result[0] += 0.012123995332696655;
                            } else {
                              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.03444837977138745;
                              } else {
                                result[0] += -0.12247816814423725;
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.020590832513732903;
                        } else {
                          result[0] += -0.16801628042170408;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                        result[0] += -0.07806807247909518;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.036051265420110994;
                        } else {
                          result[0] += 0.08004601261080144;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.11014711141835408;
                    } else {
                      result[0] += 0.008423793085660325;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                    result[0] += 0.048362848662240415;
                  } else {
                    result[0] += -0.05506025552887799;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
                        result[0] += -0.0011167477503794567;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                            result[0] += 0.08034562813381436;
                          } else {
                            result[0] += 0.007783338266411009;
                          }
                        } else {
                          result[0] += -0.05559185818728123;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                          result[0] += 0.0173680805277827;
                        } else {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.05656087384723804;
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                              result[0] += 0.016164466693364944;
                            } else {
                              result[0] += -0.07979903556735196;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.07411915801006545;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.016937317457389044;
                        } else {
                          result[0] += 0.0622396621510469;
                        }
                      } else {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                          result[0] += -0.012973155266065737;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.07560310658941528;
                          } else {
                            result[0] += 0.05603031158682231;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.037925759093334885;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.7719768400272474;
                } else {
                  result[0] += 0.07711903305841417;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                  result[0] += -0.05303568169963377;
                } else {
                  result[0] += 0.02601363718252568;
                }
              }
            }
          } else {
            result[0] += -0.05075488397807437;
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07180084846239626;
            } else {
              result[0] += -0.027965055892369167;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.07134145200115373;
            } else {
              result[0] += 0.02681749132709281;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
              result[0] += -0.024174053753404163;
            } else {
              result[0] += 0.024387591901402713;
            }
          } else {
            result[0] += 0.02799627229488388;
          }
        } else {
          if ( UNLIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += 0.07078844672454403;
              } else {
                result[0] += 0.0042127404942863605;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.033514977659107256;
                } else {
                  result[0] += 0.018454174326221425;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.07097614300925802;
                } else {
                  result[0] += -0.0022196920212484974;
                }
              }
            }
          } else {
            result[0] += 0.00958855973029558;
          }
        }
      }
    } else {
      result[0] += -0.11868476505520498;
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.003805924218591339;
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
          result[0] += 0.060106270546295894;
        } else {
          result[0] += -0.015402383264805636;
        }
      } else {
        result[0] += -0.08208514545749707;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.010509624429344654;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.010696359599320318;
                } else {
                  result[0] += -0.04883347548577695;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += -0.042094283737256044;
              } else {
                result[0] += 0.05447390380678635;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.018371771344351067;
              } else {
                result[0] += -0.03067869354319512;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.06916384561515475;
              } else {
                result[0] += -0.02523793639479868;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              result[0] += -0.0019510870270842225;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                result[0] += 0.047101870641245165;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += 0.009545647379088822;
                  } else {
                    result[0] += -0.07003600695000714;
                  }
                } else {
                  result[0] += 0.026677315291901774;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                result[0] += -0.0752877391989325;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += -0.06139446056098431;
                      } else {
                        result[0] += 0.03266512624454557;
                      }
                    } else {
                      result[0] += -0.11518811423423546;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.11177988290982198;
                    } else {
                      result[0] += -0.03467134740893248;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.018271072961093725;
                    } else {
                      result[0] += -0.09538185823035254;
                    }
                  } else {
                    result[0] += 0.05866343643323307;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.978102684020996982) ) ) {
                      result[0] += 0.02420978168917632;
                    } else {
                      result[0] += -0.057085495772311735;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
                        result[0] += -0.06423785510862294;
                      } else {
                        result[0] += 0.005208645672493294;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += 0.05044949157132425;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.605039834976196733) ) ) {
                          result[0] += -0.0963556748416311;
                        } else {
                          result[0] += -0.3726691356497871;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.03863059950945494;
                  } else {
                    result[0] += -0.09410236540690997;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += -0.0994594494768495;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.005495870720211487;
                      } else {
                        result[0] += 0.10263145280912994;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += -0.07270183958485545;
                        } else {
                          result[0] += -0.03554563192474288;
                        }
                      } else {
                        result[0] += 0.023002681670090286;
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += 0.054820520891304125;
                        } else {
                          result[0] += -0.08557390089265005;
                        }
                      } else {
                        result[0] += 0.05345253472655229;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.004885570868848233;
                    } else {
                      result[0] += 0.013647237091184912;
                    }
                  } else {
                    result[0] += -0.06154439024098516;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
          result[0] += 0.00602684547658897;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.024068881525922927;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += -0.073596731858641;
                } else {
                  result[0] += -0.15216379988600315;
                }
              }
            } else {
              result[0] += 0.03457122283876889;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += -0.058958833898280286;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.02936658711108749;
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.03456129999691059;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                        result[0] += -0.01692413098109863;
                      } else {
                        result[0] += 0.049464394969057524;
                      }
                    } else {
                      result[0] += 0.07442528636116133;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                        result[0] += -0.06572851484833137;
                      } else {
                        result[0] += 0.019008721185407387;
                      }
                    } else {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.013639940128132973;
                      } else {
                        result[0] += -0.1016343568495052;
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
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
      result[0] += 0.005440948077064163;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
          result[0] += 0.02878266544484956;
        } else {
          result[0] += -0.02770441597850992;
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          result[0] += -0.048849091487826554;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.011629241918501938;
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += -0.05795886925030361;
              } else {
                result[0] += 0.021758162575226718;
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += -0.08422894475111084;
                } else {
                  result[0] += 0.37390359434888404;
                }
              } else {
                result[0] += -0.06125248624988688;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.04972868024958829;
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)32.50000000000000711) ) ) {
                  result[0] += -0.1599250831442033;
                } else {
                  result[0] += 0.034338088318341446;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.04655098395980924;
        } else {
          result[0] += -0.014829414309598776;
        }
      } else {
        result[0] += -0.0811462556626854;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.07626164505285206;
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.04315709158791505;
                } else {
                  result[0] += 0.006369273716908865;
                }
              }
            } else {
              result[0] += 0.020832702648309494;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.003553508981728748;
            } else {
              result[0] += -0.05047285823197553;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.06425870634133737;
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.016274358509384062;
                } else {
                  result[0] += -0.04271266844290912;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.06932613606481455;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.0237605506882615;
                  } else {
                    result[0] += 0.07255894839093935;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.046200443624238224;
                    } else {
                      result[0] += -0.05852539031369858;
                    }
                  } else {
                    result[0] += -0.0886389593929581;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.05533470653424149;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += 0.06214273833587078;
                    } else {
                      result[0] += 0.015254975597398355;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                result[0] += -0.024428271957079507;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.354025125503540261) ) ) {
                  result[0] += -0.06876232661665518;
                } else {
                  result[0] += 0.011689810589915786;
                }
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.04003272901839927;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += -0.0706690912354677;
                      } else {
                        result[0] += 0.006725109475514995;
                      }
                    }
                  } else {
                    result[0] += 0.016640765243072057;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.03950043272821243;
                  } else {
                    result[0] += -0.016722606111104123;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.09403133512316603;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.10843450959509876;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.02256299277466785;
                        } else {
                          result[0] += -0.11352549258726151;
                        }
                      }
                    } else {
                      result[0] += 0.009677179839671738;
                    }
                  }
                } else {
                  result[0] += 0.0486194946006365;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += 0.06445930833417148;
            } else {
              result[0] += 0.002012114747667125;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              result[0] += -0.036596789481016315;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += 0.006369737651915864;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.057787687501640296;
                } else {
                  result[0] += -0.00269527085448985;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.09604339752689588;
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
              result[0] += -0.06153825205617386;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.013414904832397193;
                } else {
                  result[0] += 0.02514925067871378;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += 0.07346497717309085;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                      result[0] += -0.11993168261955481;
                    } else {
                      result[0] += -0.009318024900288114;
                    }
                  } else {
                    result[0] += -0.09447190490806251;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.0004102492902224137;
            } else {
              result[0] += 0.10129990127201893;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.051998598906885234;
            } else {
              result[0] += -0.055679586335184675;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.000308036804200107) ) ) {
              result[0] += -0.07577287118506616;
            } else {
              result[0] += 0.005034221003376965;
            }
          } else {
            result[0] += 0.05727229492489031;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.912734985351563388) ) ) {
                result[0] += 0.010877071794745709;
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                    result[0] += -0.0016684693348103302;
                  } else {
                    result[0] += -0.054694211273039084;
                  }
                } else {
                  result[0] += 0.060169312364186;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                result[0] += -0.09112002051398678;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)23.50000000000000355) ) ) {
                  result[0] += -0.04210827580829392;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                    result[0] += 0.2728530546103291;
                  } else {
                    result[0] += -0.013121730000580423;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                  result[0] += -0.06430749254819412;
                } else {
                  result[0] += -0.17264478527737043;
                }
              } else {
                result[0] += 0.06953771881070536;
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += -0.028736377560556987;
                  } else {
                    result[0] += 0.03466386197208686;
                  }
                } else {
                  result[0] += 0.0659026016371617;
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                      result[0] += 0.04642809946144896;
                    } else {
                      result[0] += 0.34178366716034303;
                    }
                  } else {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += -0.08847638239881395;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                          result[0] += -0.031693101483927615;
                        } else {
                          result[0] += 0.2785277891195344;
                        }
                      }
                    } else {
                      result[0] += -0.0806135585596233;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.12712441771039262;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                      result[0] += 0.02449001265222128;
                    } else {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                        result[0] += -0.11146428442530404;
                      } else {
                        result[0] += 0.07172719438174593;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.08959157534273499;
          } else {
            result[0] += -0.0930154503436588;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.00868761149796073;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.09575245617938449;
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                      result[0] += 0.03476073022369571;
                    } else {
                      result[0] += -0.07814488105448447;
                    }
                  }
                }
              } else {
                result[0] += -0.03833842325777932;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.06797265864480224;
                } else {
                  result[0] += 0.005859868785123397;
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += 0.05644210350710767;
                } else {
                  result[0] += 0.009172212798525045;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
              result[0] += -0.10456595065828421;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.74696540832519709) ) ) {
                result[0] += -0.002007024975393424;
              } else {
                result[0] += -0.0864811750942888;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.008330479149001264;
            } else {
              result[0] += 0.039828509989808934;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)20.50000000000000355) ) ) {
                    result[0] += -0.008992124139662544;
                  } else {
                    result[0] += 0.09587319167504346;
                  }
                } else {
                  result[0] += -0.021858378952773604;
                }
              } else {
                if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.009486715439253095;
                } else {
                  result[0] += -0.06973640814963457;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                result[0] += -0.06521106189131136;
              } else {
                result[0] += 0.12290217026114406;
              }
            }
          }
        }
      } else {
        result[0] += 0.007008129671711847;
      }
    }
  } else {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
          result[0] += 0.06792285519761693;
        } else {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
            result[0] += -0.005226435022698844;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.671254158020020419) ) ) {
                result[0] += -0.053136666656433906;
              } else {
                result[0] += 0.028644929442265345;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.08199150310047555;
              } else {
                result[0] += -0.019407214645630024;
              }
            }
          }
        }
      } else {
        result[0] += -0.07748991374472397;
      }
    } else {
      result[0] += -0.0008863214525235721;
    }
  }
  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.0035323278685486293;
  } else {
    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            result[0] += -0.01947774511465877;
          } else {
            result[0] += 0.013203780923404488;
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.2687106132507342) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.59331560134887873) ) ) {
                    result[0] += -0.007959397099416692;
                  } else {
                    result[0] += 0.03221908259715996;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.07783815559625362;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.445957899093628818) ) ) {
                      result[0] += 0.038626870706688285;
                    } else {
                      result[0] += -0.28501110780101097;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.12462248933134208;
                } else {
                  result[0] += -0.02057770676271894;
                }
              }
            } else {
              result[0] += 0.03549669916139905;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.06073302558903337;
              } else {
                result[0] += -0.03186947533892177;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += 0.04507774181519207;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.048252819266359465;
                } else {
                  result[0] += -0.007167161739391387;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
          result[0] += 0.009099962980140534;
        } else {
          result[0] += -0.061866250586095455;
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.04072233487649813;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
                    result[0] += -0.06755920817637022;
                  } else {
                    result[0] += 0.06584329263444462;
                  }
                } else {
                  result[0] += -0.04184411381238154;
                }
              } else {
                result[0] += -0.052344990985253054;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
                  result[0] += -0.035626568389043475;
                } else {
                  result[0] += 0.1598523528588086;
                }
              } else {
                result[0] += 0.024985023116373035;
              }
            } else {
              result[0] += -0.04058741624790988;
            }
          }
        } else {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.02156374465728013;
              } else {
                result[0] += -0.1208064158906901;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                        result[0] += 0.09068794971544271;
                      } else {
                        result[0] += -0.006114366884155398;
                      }
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.011678357989449962;
                      } else {
                        result[0] += -0.04106996365023125;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
                        result[0] += 0.10364325027664763;
                      } else {
                        result[0] += -0.027760092247168074;
                      }
                    } else {
                      result[0] += 0.09028686610383767;
                    }
                  }
                } else {
                  result[0] += -0.011202229542012714;
                }
              } else {
                result[0] += -0.008071098607817389;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.014433165551889837;
                } else {
                  result[0] += 0.13577163203311246;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.347890853881836826) ) ) {
                    result[0] += -0.061015789867502224;
                  } else {
                    result[0] += 0.04506082664711;
                  }
                } else {
                  result[0] += -0.009844643210309333;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += 0.09725481144536365;
                    } else {
                      result[0] += 0.02161033307346347;
                    }
                  } else {
                    result[0] += -0.002416231019735109;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                      result[0] += -0.09315293571326472;
                    } else {
                      result[0] += 0.037038607914011944;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.05442981795815044;
                      } else {
                        result[0] += 0.028714575511235033;
                      }
                    } else {
                      result[0] += 0.06184330423993604;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                      result[0] += 0.018674320840964025;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                        result[0] += -0.060466645909108246;
                      } else {
                        result[0] += 0.04059684042969939;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                        result[0] += 0.09134417417173035;
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                            result[0] += -0.010462450017200739;
                          } else {
                            result[0] += 0.13041209209440127;
                          }
                        } else {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.010627043387883926;
                          } else {
                            result[0] += 0.11624538915847815;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.0057629337084642305;
                    }
                  }
                } else {
                  result[0] += -0.0496743943870673;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
          result[0] += -0.07189584565509631;
        } else {
          result[0] += 0.002413458210011407;
        }
      }
    }
  }
  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.007543282981506982;
            } else {
              result[0] += -0.003990200244636448;
            }
          } else {
            result[0] += 0.04279894783244027;
          }
        } else {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
            result[0] += -0.0003255534004374032;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                result[0] += -0.026068607917743698;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.06806327645724623;
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.07187488082113992;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.0805291552683418;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                          result[0] += 0.04082193345596285;
                        } else {
                          result[0] += -0.04073591542192245;
                        }
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.040935259772857495;
                        } else {
                          result[0] += 0.01600884700609034;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.03844328495739152;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += -0.022382640418078775;
          } else {
            result[0] += -0.13896747254080571;
          }
        } else {
          if ( UNLIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += 0.004717979506975084;
            } else {
              result[0] += 0.03670672214245406;
            }
          } else {
            result[0] += 0.005319608521652297;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        result[0] += 0.09120752584416711;
      } else {
        result[0] += -0.09958156593463005;
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
          result[0] += 0.04509973115058501;
        } else {
          result[0] += -0.006879053062302666;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.04030319987760922;
            } else {
              result[0] += 0.09583909249746893;
            }
          } else {
            result[0] += -0.03572150016621575;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205872535705568183) ) ) {
            result[0] += -0.06886542562303503;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.039292328965938206;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)168.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.06337914746321809;
                } else {
                  result[0] += 0.007740337826868006;
                }
              } else {
                result[0] += -0.12894059113335257;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.04457934302521718;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.004478666739106028;
              } else {
                result[0] += 0.03884615786919146;
              }
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                  result[0] += -0.06479563222723846;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += -0.20843925498488725;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                      result[0] += 0.09091779166461122;
                    } else {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                        result[0] += 0.053545347096505096;
                      } else {
                        result[0] += -0.024743902497727583;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                  result[0] += 0.06456710995539353;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.03157564468703288;
                  } else {
                    result[0] += -0.08644206703889458;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += -0.11842161717836844;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
                          result[0] += -0.009521772804130822;
                        } else {
                          result[0] += -0.11659691521349398;
                        }
                      }
                    } else {
                      result[0] += 0.022554286606060812;
                    }
                  } else {
                    result[0] += -0.09427708250395153;
                  }
                } else {
                  result[0] += -0.04555239826702253;
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.03707371654107827;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.05351529531708407;
                      } else {
                        result[0] += -0.049106089242295724;
                      }
                    } else {
                      result[0] += 0.00013424124958534645;
                    }
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += 0.009467375568681796;
                    } else {
                      result[0] += -0.06065860303971539;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
              result[0] += -0.026830980618220802;
            } else {
              result[0] += 0.02951823492147057;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
              result[0] += -0.0018375594489240046;
            } else {
              result[0] += 0.06848848625630816;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.05440224900167528;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
              result[0] += -0.027783732886908677;
            } else {
              result[0] += 0.012389668673809104;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += -0.07453085092486837;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.06298145614423818;
            } else {
              result[0] += 0.00030981316823400554;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.938867926597595659) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.00020443378090903846;
          } else {
            result[0] += -0.046929339649996876;
          }
        } else {
          result[0] += 0.06992072213001958;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.01559721247715054;
        } else {
          result[0] += 0.051043907007065105;
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.08815076997673447;
              } else {
                result[0] += -0.019960492522454482;
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += 0.07494579153616586;
                } else {
                  result[0] += -0.04733883241919477;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += -0.026904054321235066;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                    result[0] += 0.08412601317790891;
                  } else {
                    result[0] += -0.03331356134887392;
                  }
                }
              }
            }
          } else {
            result[0] += -0.002862592352179895;
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.1061333171532468;
              } else {
                result[0] += 0.007411145532509622;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.06723869298503092;
                  } else {
                    result[0] += -0.03704518270175351;
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                      result[0] += 0.08773960196555336;
                    } else {
                      result[0] += 0.017660147129425045;
                    }
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.142630577087403232) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                          result[0] += -0.0031694062277597096;
                        } else {
                          result[0] += -0.04712497505026257;
                        }
                      } else {
                        result[0] += 0.00704131384139577;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.627130746841431552) ) ) {
                        result[0] += -0.13083920457544737;
                      } else {
                        result[0] += -0.025096306381834445;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.1033806483588974;
                } else {
                  result[0] += -0.023341531618227607;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.297559976577759233) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0021643278667909096;
              } else {
                result[0] += -0.10080678133256243;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.06745718143901056;
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                      result[0] += -0.0006343540023505826;
                    } else {
                      result[0] += -0.07514954450324482;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
                      result[0] += 0.11751967012268674;
                    } else {
                      result[0] += 0.007380545500637462;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.24012024432015286;
                      } else {
                        result[0] += -0.1061044799014629;
                      }
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.0718262389559845;
                      } else {
                        result[0] += -0.01248829180159851;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                            result[0] += -0.10796292990059726;
                          } else {
                            result[0] += 0.02418499354529792;
                          }
                        } else {
                          result[0] += 0.01463198147865773;
                        }
                      } else {
                        result[0] += 0.028777259761159513;
                      }
                    } else {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += 0.05649536761933486;
                        } else {
                          result[0] += -0.03765860352401321;
                        }
                      } else {
                        result[0] += -0.12149452917883648;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.10117757660642519;
                  } else {
                    result[0] += 0.03085621020211814;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)10.00000000000000178) ) ) {
          result[0] += -0.012386667594677131;
        } else {
          result[0] += -0.17402795258398918;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += 0.06414539144267362;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.026157257396174458;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.06809668963405846;
                } else {
                  result[0] += 0.039311508066875;
                }
              } else {
                result[0] += -0.10103374018309773;
              }
            } else {
              result[0] += 0.023234726686315425;
            }
          } else {
            result[0] += 0.08354070373561334;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.02600970861321843;
            } else {
              result[0] += 0.038326484330641994;
            }
          } else {
            result[0] += -0.05517878077802601;
          }
        } else {
          result[0] += 0.004849990586852484;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
              result[0] += -0.013185329764535965;
            } else {
              result[0] += 0.07356425405526673;
            }
          } else {
            result[0] += -0.06889737258608253;
          }
        } else {
          result[0] += 0.023260440973081054;
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.938867926597595659) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.313157320022583896) ) ) {
                result[0] += -0.005495122242802083;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += 0.021099870227251863;
                } else {
                  result[0] += -0.039293134108085666;
                }
              }
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                  result[0] += -0.019316384345686572;
                } else {
                  result[0] += -0.11462758098472392;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                  result[0] += 0.012404894937112328;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.026494079187492255;
                    } else {
                      result[0] += -0.0029993359419988683;
                    }
                  } else {
                    result[0] += 0.03905647927820226;
                  }
                }
              }
            }
          } else {
            result[0] += 0.06857625414830516;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.01488412052561991;
          } else {
            result[0] += 0.048163831931796325;
          }
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.009768406404770788;
              } else {
                result[0] += 0.019552900152938965;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                      result[0] += -0.023149572984563868;
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 8.189660341665287e-05;
                      } else {
                        result[0] += -0.08500836336667345;
                      }
                    }
                  } else {
                    result[0] += -0.08157014244977459;
                  }
                } else {
                  result[0] += 0.0003457520021071702;
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += 0.057481320758635504;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.744781017303467685) ) ) {
                      result[0] += -0.0556898868073935;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += -0.00737684695552154;
                      } else {
                        result[0] += 0.04206150085172981;
                      }
                    }
                  }
                } else {
                  result[0] += -0.002379114382237601;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.01950019459316183;
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.05772950098544083;
                } else {
                  result[0] += 0.005140286039782059;
                }
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.114721298217775214) ) ) {
                  result[0] += 0.012920394998744426;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.06103564780088175;
                  } else {
                    result[0] += -0.00942517365412448;
                  }
                }
              } else {
                result[0] += 0.014546017875478993;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.16614955473984794;
                } else {
                  result[0] += 0.037584447029491125;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.0821992741948073;
                    } else {
                      result[0] += -0.008115389405928526;
                    }
                  } else {
                    result[0] += -0.016945280733105507;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    result[0] += -0.08198282491097741;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += -0.03619163051304164;
                    } else {
                      result[0] += 0.017552496185629218;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.0007797594777997029;
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.008464498186571114;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                    result[0] += -0.030370791302418394;
                  } else {
                    result[0] += 0.008135487632554901;
                  }
                } else {
                  result[0] += -0.034886592474111924;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                    result[0] += 0.014749756650281096;
                  } else {
                    result[0] += -0.012046319213407684;
                  }
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.008542741565265027;
                    } else {
                      result[0] += 0.05095279424698243;
                    }
                  } else {
                    result[0] += -0.07530200548998084;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.05031910311398319;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
          result[0] += -0.030206425719858584;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            result[0] += -0.08181447476458488;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.0786288078084638;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
                result[0] += 0.09529984708849473;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.052907082934119276;
                } else {
                  result[0] += 0.019765265122750425;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -0.04861279122350594;
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += 0.020869557479779244;
          } else {
            result[0] += -0.0981188632188976;
          }
        } else {
          result[0] += 0.10270647452910853;
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.0373526130619777;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.06923282903787245;
          } else {
            result[0] += 0.005998651919166413;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.87008237838745206) ) ) {
        result[0] += 0.0007352753762697192;
      } else {
        result[0] += 0.019453395188844314;
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.297559976577759233) ) ) {
          result[0] += 0.05751555154636732;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0025921149835779734;
                } else {
                  result[0] += 0.0743502279052939;
                }
              } else {
                result[0] += -0.03421283498909328;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.87254357337951749) ) ) {
                result[0] += 0.1582336768841318;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
                  result[0] += -0.0809994445181288;
                } else {
                  result[0] += 0.04532704812574123;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.246492147445679599) ) ) {
              result[0] += -0.07386890600413203;
            } else {
              result[0] += 0.06449235806189338;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.994492053985595925) ) ) {
            result[0] += -0.07163178603668316;
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                  result[0] += 0.01019030758968504;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                              result[0] += -0.09549978002392895;
                            } else {
                              result[0] += 0.07890557341719642;
                            }
                          } else {
                            result[0] += 0.04318745686563946;
                          }
                        } else {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                              result[0] += -0.034979001028895625;
                            } else {
                              result[0] += 0.13981876474891042;
                            }
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
                              result[0] += 0.023104403952848726;
                            } else {
                              result[0] += -0.10133615382819557;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.010570266929548284;
                        } else {
                          result[0] += 0.07853633031997331;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.05820550765623472;
                          } else {
                            result[0] += -0.027502247208519043;
                          }
                        } else {
                          result[0] += 0.04946738985243868;
                        }
                      } else {
                        result[0] += -0.004791543830462783;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.25930547714233576) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.002530496783121346;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                            result[0] += 0.03589722490966453;
                          } else {
                            result[0] += -0.024667932853856406;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += 0.008878470433436996;
                        } else {
                          result[0] += -0.022045442566133427;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.020187989754848305;
                          } else {
                            result[0] += -0.03882927459604819;
                          }
                        } else {
                          result[0] += -0.15743357926038204;
                        }
                      } else {
                        result[0] += 0.03561042424364319;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.04695697374724317;
                } else {
                  result[0] += -0.005522872312083685;
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.06553852179782275;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.0068812156396686865;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.41262340545654475) ) ) {
                    result[0] += -0.07791414072414601;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                          result[0] += 0.029689763623580057;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.09333488434255544;
                            } else {
                              result[0] += -0.04574962193582239;
                            }
                          } else {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += -0.045217738187184636;
                            } else {
                              result[0] += 0.009786779942360631;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.038870687356570344;
                      }
                    } else {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += 0.02032431881573271;
                        } else {
                          result[0] += -0.11578956662031588;
                        }
                      } else {
                        result[0] += -0.11199269626927377;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06238214019801445;
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.03495483397281595;
              } else {
                result[0] += 0.06476632422013227;
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.07654534317102914;
                  } else {
                    result[0] += -0.017123881858415776;
                  }
                } else {
                  result[0] += -0.03182915940782526;
                }
              } else {
                result[0] += 0.03389821031038754;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.11653209282708728;
              } else {
                result[0] += 0.026991838257202267;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -0.04588438659640938;
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += 0.01931367264169218;
          } else {
            result[0] += -0.09515303120157284;
          }
        } else {
          result[0] += 0.09753215594663851;
        }
      } else {
        result[0] += 0.0058738370603313335;
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += 0.00022514437715628581;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.09660939555448428;
            } else {
              result[0] += 0.021017039409001095;
            }
          }
        } else {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.041301490342336766;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.09132049386101766;
                } else {
                  result[0] += -0.01718537889271626;
                }
              } else {
                result[0] += 0.031835765451232324;
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.0037728058945121212;
            } else {
              result[0] += -0.018684825769592888;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.015010793352722526;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
            result[0] += 0.07948856218867437;
          } else {
            result[0] += 0.033898637648531765;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0013441984696336978;
          } else {
            result[0] += -0.008271768430564269;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                result[0] += -0.012797581666422331;
              } else {
                result[0] += -0.09612757298344794;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.08025350299941938;
                  } else {
                    result[0] += 0.015762390114248896;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.03881922623208408;
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      result[0] += -0.0070786756181767845;
                    } else {
                      result[0] += 0.038030432200050174;
                    }
                  }
                }
              } else {
                result[0] += -0.05665764724302624;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.07850111696499264;
            } else {
              result[0] += -0.03331186422389911;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.07241751870924792;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += 0.07166836314497967;
                  } else {
                    result[0] += -0.006146592799573423;
                  }
                } else {
                  result[0] += -0.015132429316346627;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.051070411122119144;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.046630037542473725;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += -0.09488257157314478;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += -0.034890693100789454;
                      } else {
                        result[0] += 0.04710078446029477;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0018297593742651348;
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                  result[0] += -0.06624710785478254;
                } else {
                  result[0] += -0.017575167590542673;
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.297559976577759233) ) ) {
                    result[0] += 0.048736792250938826;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.034722000407188376;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.06908550749745812;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.0257444540501874;
                        } else {
                          result[0] += -0.1862420000025421;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.046382480389883164;
                  } else {
                    result[0] += 0.04604471219405035;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.452637434005737749) ) ) {
                  result[0] += -0.04775065340485544;
                } else {
                  result[0] += 0.005960808037202229;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.007662685564016631;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.01919594737627904;
                  } else {
                    result[0] += 0.015220643006600086;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.002204387019704145;
                  } else {
                    result[0] += -0.05247939272364646;
                  }
                }
              } else {
                result[0] += -0.04001750495125375;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                  result[0] += 0.013256422173688881;
                } else {
                  result[0] += -0.0121049556701354;
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.029181598804351797;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.05042736359799854;
                  } else {
                    result[0] += -0.002121820754832946;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
      result[0] += -0.1278170901049748;
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.059144337907989064;
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.028455294878696226;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                result[0] += -0.0899440187825679;
              } else {
                result[0] += 0.023083609718303318;
              }
            }
          } else {
            result[0] += 0.01877505003321144;
          }
        }
      } else {
        result[0] += 0.004640681747304543;
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.313157320022583896) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += -0.012692708475671007;
              } else {
                result[0] += 0.0231321880337659;
              }
            } else {
              result[0] += -0.007994966120623392;
            }
          } else {
            result[0] += 0.018132374002784397;
          }
        } else {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                result[0] += -0.01644072264317463;
              } else {
                result[0] += -0.05650972740572743;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += -0.06981038082262919;
                } else {
                  result[0] += -0.014441888126156058;
                }
              } else {
                result[0] += 0.030845322462845343;
              }
            }
          } else {
            result[0] += -0.00043218230205156353;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
          result[0] += -0.00025604854929521845;
        } else {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += 0.1068780384615027;
            } else {
              result[0] += 0.0339482795511461;
            }
          } else {
            result[0] += 0.023365098786925737;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0006434755057003401;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.0036647808803215195;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.014927941538864798;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.08109338235504039;
                } else {
                  result[0] += -0.03921945936027368;
                }
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.015697922631337912;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                  result[0] += 0.053992899234664674;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.18372630039739785;
                  } else {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                        result[0] += 0.6479275685250818;
                      } else {
                        result[0] += 1.7909833765367857;
                      }
                    } else {
                      result[0] += 0.02030043684119439;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.807813167572023261) ) ) {
              result[0] += 0.028418435601769043;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
                result[0] += -0.008286893295664169;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  result[0] += -0.012022400547900597;
                } else {
                  result[0] += 0.0526673290234293;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.001397046665111909;
              } else {
                result[0] += -0.09001316378610708;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.61392831802368342) ) ) {
                result[0] += -0.06796606073021537;
              } else {
                result[0] += -0.021893472070666234;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.03279957351712654;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                  result[0] += 0.0252763986023972;
                } else {
                  result[0] += -0.06581922597092392;
                }
              }
            } else {
              result[0] += -0.048456783955603405;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.028522923757822262;
              } else {
                result[0] += -0.021264535503629715;
              }
            } else {
              result[0] += 0.011598075898622695;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
      result[0] += -0.12639008387737502;
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
            result[0] += 0.135422044355989;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.01723898784860278;
            } else {
              result[0] += 0.08611797681492867;
            }
          }
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.019338057749067508;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.03604437972453237;
              } else {
                result[0] += 0.05642419968037469;
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.012983762831842598;
                } else {
                  result[0] += 0.07299404841677336;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.015906904855711414;
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.09356329611618164;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += 0.02935090626996773;
                    } else {
                      result[0] += -0.03567643233652971;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.003781980303189119;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.667095184326172763) ) ) {
                result[0] += 0.009978430209767422;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0012582454853014686;
                } else {
                  result[0] += 1.8157971194023244;
                }
              }
            } else {
              result[0] += -0.07546673380553062;
            }
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.058684162297014446;
            } else {
              result[0] += -0.04144233318680644;
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.00026993219581381597;
              } else {
                result[0] += 0.057598911170014824;
              }
            } else {
              result[0] += -0.0296463275463449;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.921924352645874468) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.08971268695472638;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.017080156479650718;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.022573445485766487;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                        result[0] += -0.00045812776845383075;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.09018293666854403;
                          } else {
                            result[0] += 0.02862878972370045;
                          }
                        } else {
                          result[0] += 4.0398643632455054e-05;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                        result[0] += -0.018820481998150555;
                      } else {
                        result[0] += 0.019085692784606062;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.888949394226075995) ) ) {
                          result[0] += -0.11484660278911946;
                        } else {
                          result[0] += 0.03337734190086604;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35709619522094904) ) ) {
                          result[0] += 0.1174086136758133;
                        } else {
                          result[0] += 0.030016859932477654;
                        }
                      }
                    } else {
                      result[0] += -0.0009776825080825755;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += -0.02169832499558838;
            } else {
              result[0] += -0.13520362274561967;
            }
          }
        } else {
          result[0] += -0.050365759819381496;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.015770431009320806;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += -0.19796986435153097;
                } else {
                  result[0] += -0.037586490294367336;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.05643576481111462;
              } else {
                result[0] += -0.0734310525334272;
              }
            }
          } else {
            result[0] += 0.03541297174172711;
          }
        } else {
          result[0] += 0.02221793928765843;
        }
      }
    } else {
      result[0] += -0.0018564651471955002;
    }
  } else {
    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += 0.006116032026145716;
                      } else {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += 0.04751384930716457;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                            result[0] += 0.09367200074066319;
                          } else {
                            result[0] += -0.019236969789533532;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.0022035451902806136;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                          result[0] += -0.0661003213401195;
                        } else {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                              result[0] += 0.04197277578360247;
                            } else {
                              result[0] += 0.005537290133807794;
                            }
                          } else {
                            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                              result[0] += -0.02486175059701302;
                            } else {
                              result[0] += 0.07809863504597749;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.06421788851459363;
                      } else {
                        result[0] += 0.03154840987467809;
                      }
                    } else {
                      result[0] += 0.016121546018835486;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                    result[0] += -0.08103225374634883;
                  } else {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                        result[0] += 0.09673815801812255;
                      } else {
                        result[0] += -0.08201622957329105;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.16719671385194554;
                      } else {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.16383361201041544;
                        } else {
                          result[0] += -0.035693401228842285;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.06470329339334972;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.042435407638550693) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.02900028228759943) ) ) {
                  result[0] += -0.054976067446474666;
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.1675126655654802;
                    } else {
                      result[0] += -0.12635516272575764;
                    }
                  } else {
                    result[0] += -0.05633603452287969;
                  }
                }
              } else {
                result[0] += 0.0002719196457977469;
              }
            }
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += -0.029649933421314657;
                } else {
                  result[0] += -0.071984225007524;
                }
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.06102862547884422;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.02594850948209261;
                  } else {
                    result[0] += 0.02235476930380378;
                  }
                }
              }
            } else {
              result[0] += 0.003339328573590849;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.06402600699587117;
            } else {
              result[0] += 0.01456673852706984;
            }
          } else {
            result[0] += 0.014605579892840795;
          }
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
            result[0] += 0.033786529560337814;
          } else {
            result[0] += -0.04361387072995185;
          }
        } else {
          result[0] += 0.03728776537148822;
        }
      }
    } else {
      result[0] += -0.13000171268922386;
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.07033832434531842;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += 0.014618242292063844;
            } else {
              result[0] += -0.001666414688364298;
            }
          }
        } else {
          result[0] += -0.05094441627472876;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
          result[0] += -0.018507090374639792;
        } else {
          result[0] += 0.021117724623073885;
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.500000000000000888) ) ) {
        result[0] += -0.0017325164018324292;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.65906000137329146) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
            result[0] += 0.3579501851470268;
          } else {
            result[0] += -0.05076155350362728;
          }
        } else {
          result[0] += 0.3815822965585696;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += 0.015375783085197936;
          } else {
            result[0] += -0.09142564583208407;
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
              result[0] += 0.05627126521651171;
            } else {
              result[0] += -0.08138501217827084;
            }
          } else {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
              result[0] += 0.025798882417187665;
            } else {
              result[0] += 0.11858234999926545;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.053950271042929224;
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.02231328925349038;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.20695014481818874;
                } else {
                  result[0] += 0.022838397044186692;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.001443811251509116;
                } else {
                  result[0] += -0.07070348266882061;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                  result[0] += -0.004932038862917331;
                } else {
                  result[0] += 0.0951898571235362;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                  result[0] += 0.027498113113748557;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += 0.008702242180838453;
                  } else {
                    result[0] += -0.10333018665270094;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                      result[0] += -0.08286491450287616;
                    } else {
                      result[0] += 0.002465642929238344;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.08323932421488076;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.347890853881836826) ) ) {
                          result[0] += -0.015939944330041613;
                        } else {
                          result[0] += 0.0530088885710014;
                        }
                      }
                    } else {
                      result[0] += 0.03473357188896358;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += 0.016750943440070575;
                  } else {
                    result[0] += -0.026872816864293072;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.0791519176277124;
              } else {
                result[0] += 0.0030514775296855176;
              }
            } else {
              result[0] += -0.08178474895274766;
            }
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.0642813999680531;
                  } else {
                    result[0] += -0.07107705680661384;
                  }
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.017685512687512207;
                  } else {
                    result[0] += 0.07697632360648082;
                  }
                }
              } else {
                result[0] += -0.05522036169541636;
              }
            } else {
              if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.08022492064108698;
                    } else {
                      result[0] += -0.04823103180154742;
                    }
                  } else {
                    result[0] += -0.046048266858100095;
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                    result[0] += -0.0951175896601422;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.042435407638550693) ) ) {
                      result[0] += 0.765114548658855;
                    } else {
                      result[0] += 0.0518817442709442;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                      result[0] += -0.038144722589448454;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.015760194587164576;
                        } else {
                          result[0] += 0.09423926491242539;
                        }
                      } else {
                        result[0] += -0.030738068334509674;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                        result[0] += 0.0532894107828516;
                      } else {
                        result[0] += -0.0014434423610134132;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.0028694758781054926;
                      } else {
                        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                            result[0] += -0.01978368806076922;
                          } else {
                            result[0] += -0.08051264843815117;
                          }
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.05275830287834419;
                          } else {
                            result[0] += -0.08675510510207252;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                    result[0] += -0.012764849578520128;
                  } else {
                    result[0] += 0.1475506064459951;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.12472587136239047;
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.022860211558682383;
        } else {
          result[0] += 0.13760183599158296;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.003272205594894496;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.018078557757163702;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.01767395580622352;
                      } else {
                        result[0] += -0.11947034354429927;
                      }
                    }
                  }
                } else {
                  result[0] += -0.05366501923181782;
                }
              } else {
                result[0] += 0.021627040280702458;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.09397684000337043;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.799065828323365146) ) ) {
                        result[0] += -0.17068637609043882;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                          result[0] += 0.00039422886464804505;
                        } else {
                          result[0] += -0.05858737106972987;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                      result[0] += 0.06803956683378773;
                    } else {
                      result[0] += -0.06129032962458311;
                    }
                  }
                } else {
                  result[0] += 0.03834257452986561;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.07643061022079288;
                    } else {
                      result[0] += -0.0389349735926792;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
                      result[0] += -0.13015370095567602;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                        result[0] += 0.010174612606409124;
                      } else {
                        result[0] += -0.04321282504267311;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.13207148067992328;
                      } else {
                        result[0] += 0.014161493108861273;
                      }
                    } else {
                      result[0] += 0.043110638616243036;
                    }
                  } else {
                    result[0] += 0.00023470900280526366;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.082005289289777;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.481347560882569248) ) ) {
                result[0] += -0.07311092570389553;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                  result[0] += 0.03476414713836637;
                } else {
                  result[0] += -0.03094084954532584;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.005700309738653641;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.08351559507431504;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.08030774119770943;
                } else {
                  result[0] += 0.029272660959162772;
                }
              }
            }
          } else {
            result[0] += -0.05863741550969052;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.0252115760897717;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.03671303869861793;
                    } else {
                      result[0] += -0.12923112048231605;
                    }
                  } else {
                    result[0] += -0.10075454026600511;
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)23.50000000000000355) ) ) {
                    result[0] += -0.0674944452019991;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                      result[0] += 0.16077798136097932;
                    } else {
                      result[0] += -0.08343174166882315;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                  result[0] += -0.07564119112495125;
                } else {
                  result[0] += 0.010190974570412457;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += -0.09252249734564161;
              } else {
                result[0] += 0.028290272862819488;
              }
            }
          } else {
            result[0] += -0.09127211320016039;
          }
        }
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.06409598360529628;
              } else {
                result[0] += -0.008925380805079361;
              }
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.0016127961886259633;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.002837981413289845;
                } else {
                  result[0] += -0.041859021635077605;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.05284856275744193;
            } else {
              result[0] += -0.011267777317457978;
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.0584630944504609;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += 0.056745194877499405;
            } else {
              result[0] += 0.00204494851988812;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          result[0] += 0.014316498071214184;
        } else {
          result[0] += -0.0884480930812388;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.020433258019388067;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.12083605441989816;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.10333013850199423;
            } else {
              result[0] += -0.030354978723106437;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        result[0] += 0.005688529846092603;
      } else {
        result[0] += -0.018874543748862636;
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                result[0] += 0.016156262989129765;
              } else {
                result[0] += -0.0005771249806407163;
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                  result[0] += -0.016771910500156913;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += -0.0022735087493644464;
                  } else {
                    result[0] += -0.08498048657462665;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.054540167916726036;
                } else {
                  result[0] += -0.002263330931828713;
                }
              }
            }
          } else {
            result[0] += -0.06872401576541809;
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
              result[0] += 0.0012604847229206384;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
                result[0] += 0.04378528274060547;
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0009634083980064952;
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0589272369577899;
                  } else {
                    result[0] += -0.009680784634423052;
                  }
                }
              }
            }
          } else {
            result[0] += 0.004735666793310101;
          }
        }
      } else {
        result[0] += 0.04024678671144297;
      }
    } else {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)10.00000000000000178) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.08707292984357236;
          } else {
            result[0] += -0.023681083297735814;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.059293270111084873) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.13056906998455758;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.07146248288891317;
                  } else {
                    result[0] += 0.05570290181698916;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                    result[0] += 0.03962860887393141;
                  } else {
                    result[0] += 0.16006775065398127;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.1281468415018839;
                } else {
                  result[0] += -0.012029860993557795;
                }
              } else {
                result[0] += -0.011814378271278963;
              }
            }
          } else {
            result[0] += -0.380547214830058;
          }
        }
      } else {
        result[0] += -0.18031378772972287;
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          result[0] += 0.013759310487450447;
        } else {
          result[0] += -0.08590893172402533;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.01806542303438134;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.12404393306931555;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.0928221955883327;
            } else {
              result[0] += -0.030929887844157084;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.010167064226559486;
              } else {
                result[0] += 0.01558644390716619;
              }
            } else {
              result[0] += -0.0401736774822453;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += -0.06801264695448335;
              } else {
                result[0] += 0.009246613304060002;
              }
            } else {
              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.023308877594172297;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.11161272575041468;
                      } else {
                        result[0] += 0.03517408552324241;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                        result[0] += -0.09103317554101326;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.06933583919628651;
                        } else {
                          result[0] += -0.0025386689445778174;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.01036654614805896;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.347890853881836826) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                          result[0] += 0.06360394163046013;
                        } else {
                          result[0] += -0.037773033945226644;
                        }
                      } else {
                        result[0] += 0.038899154069565206;
                      }
                    } else {
                      result[0] += 0.006169492084120164;
                    }
                  } else {
                    result[0] += 0.035444283949769034;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                    result[0] += 0.003367367178483225;
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                      result[0] += 0.028915242815062178;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.141444921493531162) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.08094183695702606;
                        } else {
                          result[0] += 0.12302797245673713;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += 0.2683995824124444;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                            result[0] += 0.34318153618156105;
                          } else {
                            result[0] += 1.4321315005806943;
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.03393472101914347;
                }
              } else {
                result[0] += -0.05172117339201582;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                result[0] += -0.042035789225337024;
              } else {
                result[0] += 0.05787046503051433;
              }
            }
          } else {
            result[0] += 0.02386535689489693;
          }
        }
      } else {
        result[0] += -0.024450845804312324;
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)240501391360.0000305) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.12353437009053317;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0008753235132402366;
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                  result[0] += -0.13282468808730186;
                } else {
                  result[0] += -0.002600089657099477;
                }
              } else {
                result[0] += -0.06881257188517799;
              }
            }
          }
        } else {
          result[0] += -0.056004926256600474;
        }
      } else {
        result[0] += 0.01529932523347871;
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
            result[0] += 0.00029929115574769743;
          } else {
            result[0] += -0.06333495262350851;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.06151843520454743;
              } else {
                result[0] += -0.019227805603100574;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                result[0] += -0.0018630380516819556;
              } else {
                result[0] += -0.04761807946924072;
              }
            }
          } else {
            result[0] += 0.005745884316990216;
          }
        }
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
            result[0] += -0.002617411718832668;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.938988685607911933) ) ) {
              result[0] += 0.04091672925136758;
            } else {
              result[0] += -0.053430958468946355;
            }
          }
        } else {
          result[0] += 0.0037109930469060417;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          result[0] += 0.012632455306971086;
        } else {
          result[0] += -0.08314709304847082;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
            result[0] += 0.04935209170900148;
          } else {
            result[0] += -0.06892493409538523;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.10560946056799612;
          } else {
            result[0] += 0.024219571188121722;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += -0.051675961224150524;
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.024914300649820916;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.16605266391392995;
              } else {
                result[0] += 0.01674879076863572;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.0023166201458556775;
              } else {
                result[0] += -0.06873687859713241;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.011440715631101916;
                } else {
                  result[0] += -0.05438383416114587;
                }
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.13048316113377773;
                } else {
                  result[0] += 0.02470599949191054;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
              result[0] += 0.09115882594945274;
            } else {
              result[0] += 0.012744538315179285;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
            result[0] += -0.005011970613547574;
          } else {
            result[0] += -0.06870760700703864;
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                    result[0] += -0.05193845053128572;
                  } else {
                    result[0] += 0.09718000548415134;
                  }
                } else {
                  result[0] += -0.06742083693534892;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.015296124920492558;
                } else {
                  result[0] += 0.06988732184983358;
                }
              }
            } else {
              result[0] += -0.05510288822360848;
            }
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.05950699242732425;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                    result[0] += 0.06005440383666902;
                  } else {
                    result[0] += -0.04978253148383514;
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += -0.09291754762116179;
                } else {
                  result[0] += 0.12279190999194463;
                }
              }
            } else {
              if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += 0.009671768512309347;
                      } else {
                        result[0] += -0.06691344428701906;
                      }
                    } else {
                      result[0] += 0.013575483318245133;
                    }
                  } else {
                    result[0] += -0.056407372102180234;
                  }
                } else {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)12.50000000000000178) ) ) {
                    result[0] += -0.025519075265359226;
                  } else {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += -0.0928198312215713;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.3910219150243477;
                      } else {
                        result[0] += 1.0978737229832445;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.12147074480732328;
                    } else {
                      result[0] += 0.01534986262011654;
                    }
                  } else {
                    result[0] += -0.011331073069509522;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                      result[0] += -0.07413253934178085;
                    } else {
                      result[0] += 0.06369286145448354;
                    }
                  } else {
                    result[0] += -0.03800354087925581;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
      result[0] += -0.001699543971137375;
    } else {
      result[0] += 0.0018482471995557863;
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2202.000000000000455) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
                result[0] += -0.017706404987651275;
              } else {
                result[0] += 0.049928585030569336;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.09203972653220491;
              } else {
                result[0] += 0.028070618443980778;
              }
            }
          } else {
            result[0] += 0.050148949531394094;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
              result[0] += 0.06038175684249084;
            } else {
              result[0] += -0.004148188739984765;
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.015715738293805178;
                } else {
                  result[0] += -0.08522026917265932;
                }
              } else {
                result[0] += -0.0804971469573674;
              }
            } else {
              result[0] += 0.08397285807516269;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
          result[0] += -0.017728481302650683;
        } else {
          result[0] += -0.0609938785340229;
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
          result[0] += -0.019321009682267618;
        } else {
          result[0] += -0.14529491733033004;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.215905904769898349) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                    result[0] += 0.09260903955560638;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.10129095022199497;
                    } else {
                      result[0] += -0.11842549238361143;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.08421728232967626;
                    } else {
                      result[0] += 0.07133204844299644;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += 0.009453684856669229;
                    } else {
                      result[0] += 0.08553743247966145;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.11864673667315702;
                } else {
                  result[0] += 0.07331406976895576;
                }
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)13.50000000000000178) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.11613133344490226;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += -0.0858573301368658;
                      } else {
                        result[0] += 0.16172017282116527;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                      result[0] += -0.05841951349516411;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                        result[0] += 0.07812963630334342;
                      } else {
                        result[0] += -0.09002986664518024;
                      }
                    }
                  }
                } else {
                  result[0] += 0.07001218690687286;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                    result[0] += -0.0365798782365356;
                  } else {
                    result[0] += 0.29277002800876584;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.816582441329956943) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
                            result[0] += -0.17171072734181347;
                          } else {
                            result[0] += 0.045772841447049434;
                          }
                        } else {
                          result[0] += 0.08956353879005269;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                          result[0] += -0.06277376937791242;
                        } else {
                          result[0] += 0.06297398012263095;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                          result[0] += 0.12264980515526128;
                        } else {
                          result[0] += -0.09794378003464813;
                        }
                      } else {
                        result[0] += -0.2266855257810325;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
                          result[0] += 0.08615230534878603;
                        } else {
                          result[0] += -0.126211846647174;
                        }
                      } else {
                        result[0] += -0.11854679703110352;
                      }
                    } else {
                      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.09355160752602326;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                          result[0] += -0.016622569892267493;
                        } else {
                          result[0] += 0.3395873978723202;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                  result[0] += -0.037743204563218474;
                } else {
                  result[0] += -0.14784610018312208;
                }
              } else {
                result[0] += -0.013127796820664448;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += -0.07230276475340823;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.034678910478912875;
                  } else {
                    result[0] += 0.07968485012955664;
                  }
                }
              } else {
                result[0] += -0.02347733897260372;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.000308036804200107) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.0413207681932552;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += -0.06033287487452341;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                    result[0] += 0.04318174154632981;
                  } else {
                    result[0] += -0.03696248797321347;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                result[0] += -0.04593356121422182;
              } else {
                result[0] += 0.11151281565338221;
              }
            }
          } else {
            result[0] += 0.06318078262472109;
          }
        }
      }
    }
  }
}

