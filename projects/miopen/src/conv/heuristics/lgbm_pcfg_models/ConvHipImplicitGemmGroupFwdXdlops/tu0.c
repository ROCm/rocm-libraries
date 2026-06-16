
#include "header.h"

void predict_unit0(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.1906864142001933;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          result[0] += 0.0008740527460765329;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += -0.09581312391859245;
          } else {
            result[0] += 0.0785227603247226;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.1319425417102933;
        } else {
          result[0] += 0.01634372892892846;
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.13699136114291685;
          } else {
            result[0] += 0.07907473779862868;
          }
        } else {
          result[0] += -0.16658870248997473;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.928530216217041904) ) ) {
                      result[0] += 0.1322290250625882;
                    } else {
                      result[0] += 0.03407751970160122;
                    }
                  } else {
                    result[0] += 0.023625675410560678;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += 0.09709994201978951;
                    } else {
                      result[0] += 0.0004667674303950334;
                    }
                  } else {
                    result[0] += -0.10699784701868179;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
                    result[0] += -0.025287889780776902;
                  } else {
                    result[0] += 0.12434765813128577;
                  }
                } else {
                  result[0] += -0.08906417937085875;
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.08207526765125976;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.03067994110901423;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
                    result[0] += -0.1045582544649275;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.09917860433104284;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.059071454987833685;
                      } else {
                        result[0] += -0.04536816732672207;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.04627513645173963;
            } else {
              result[0] += 0.10912708870869946;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                result[0] += 0.12447827410077301;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                  result[0] += 0.08642158303493555;
                } else {
                  result[0] += -0.03715942979569684;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += 0.04297428734815105;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.10915502779023407;
                } else {
                  result[0] += -0.04973785982644773;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                    result[0] += 0.08978825852878541;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.028335248158798794;
                    } else {
                      result[0] += 0.08727093041795171;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += 0.044564232665320225;
                      } else {
                        result[0] += 0.11712927287380759;
                      }
                    } else {
                      result[0] += 0.031220228800237754;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                      result[0] += 0.07633134810457172;
                    } else {
                      result[0] += 0.13864956462598474;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.10204701368401803;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.08158739696820644;
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.11477466864357436;
                    } else {
                      result[0] += 0.026924990693221018;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.03550784233867249;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.09482733186173613;
            } else {
              result[0] += 0.004982724294955492;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.06751922310607127;
            } else {
              result[0] += 0.11887900992340286;
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.1665342829006858;
            } else {
              result[0] += -0.08036085142700394;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              result[0] += -0.07789279400520324;
            } else {
              result[0] += 0.011482006337112357;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.16831637132093935;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
            result[0] += -0.05701256726121687;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
              result[0] += -0.0434108908358726;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.005565984159420561;
              } else {
                result[0] += 0.12081452029099156;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
            result[0] += -0.17430488326133897;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
                result[0] += -0.07367401302917087;
              } else {
                result[0] += 0.10661125976118511;
              }
            } else {
              result[0] += -0.09611527475138831;
            }
          }
        } else {
          result[0] += -0.19616235750798028;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.16921706565745745;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.07168124265694163;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.09298416908418955;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.011039471865730968;
              } else {
                result[0] += -0.11936068299335151;
              }
            } else {
              result[0] += -0.10609743695904868;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.14187611391744961;
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.04601829541579998;
          } else {
            result[0] += -0.13247938321338823;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
              if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.08237366820685754;
              } else {
                result[0] += 0.05911152605707977;
              }
            } else {
              result[0] += 0.015459974958423878;
            }
          } else {
            if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  result[0] += 0.10218192024473616;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
                    result[0] += 0.05018659066140353;
                  } else {
                    result[0] += -0.07740115018033832;
                  }
                }
              } else {
                result[0] += -0.007424097297992545;
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.009596124670514508;
                } else {
                  result[0] += -0.06755219973820364;
                }
              } else {
                result[0] += -0.10059741118155513;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
                  result[0] += 0.036288676061513085;
                } else {
                  result[0] += -0.0884733740536022;
                }
              } else {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.05237459635732027;
                    } else {
                      result[0] += -0.044397523924367216;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
                      result[0] += -0.011282534425994608;
                    } else {
                      result[0] += -0.07352189811330666;
                    }
                  }
                } else {
                  result[0] += 0.05917116605817353;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
                result[0] += -0.05100767259516729;
              } else {
                result[0] += -0.12087462635398266;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += -0.03809751846803565;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                    result[0] += 0.005071073022217535;
                  } else {
                    result[0] += 0.07502197638044811;
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.13748854158820342;
                  } else {
                    result[0] += 0.004345081963932362;
                  }
                }
              }
            } else {
              if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.058171799209693204;
              } else {
                result[0] += 0.005906289502633461;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.05373825924271647;
              } else {
                result[0] += -0.011477410812103952;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.138696432113648349) ) ) {
                result[0] += -0.018625343372454663;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.11485448946926125;
                } else {
                  result[0] += -0.05547798287026115;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0040597992858310205;
            } else {
              result[0] += 0.09217521974290994;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += 0.03518550237961766;
              } else {
                result[0] += -0.03898737199141329;
              }
            } else {
              result[0] += -0.09709276026277429;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.490982532501221591) ) ) {
                result[0] += 0.03152303751121786;
              } else {
                result[0] += -0.09277908201486164;
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.668153762817383701) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.0653800045297304;
                    } else {
                      result[0] += -0.01652398208371551;
                    }
                  } else {
                    result[0] += 0.06763921409851714;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.07794419985925367;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.02888063351330641;
                        } else {
                          result[0] += 0.0787765579559438;
                        }
                      }
                    } else {
                      result[0] += 0.08658146305744804;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.010314441162905606;
                    } else {
                      result[0] += 0.12774740451878122;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.08795841853074136;
                } else {
                  result[0] += -0.03630275778727685;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
          result[0] += -0.13472446739700447;
        } else {
          result[0] += -0.06179291452255007;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
          result[0] += -0.11457268584975214;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
            result[0] += -0.08099987411710441;
          } else {
            result[0] += 0.08224329704691967;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += 0.14851441372810548;
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07314449942890731;
            } else {
              result[0] += 0.003758953044564885;
            }
          } else {
            result[0] += -0.11755278365253169;
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06472032934168608;
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.06566767814363429;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.02904393624443169;
                } else {
                  result[0] += 0.030874268887483444;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.11613311930971922;
            } else {
              result[0] += -0.06979033747854356;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.08253488844061385;
        } else {
          result[0] += -0.0313968228216169;
        }
      } else {
        result[0] += -0.12286823410953909;
      }
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.09712861066238915;
              } else {
                result[0] += -0.032076606882307394;
              }
            } else {
              result[0] += -0.0031495875764135047;
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                  result[0] += 0.038194065058902965;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.015879219463100596;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                      result[0] += 0.020668465745367084;
                    } else {
                      result[0] += -0.07468028464252877;
                    }
                  }
                }
              } else {
                result[0] += 0.051645133933022606;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                      result[0] += 0.0736220449099328;
                    } else {
                      result[0] += -0.005056457203871006;
                    }
                  } else {
                    result[0] += -0.014321318947956865;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.019530803635909588;
                  } else {
                    result[0] += -0.0663836027094094;
                  }
                }
              } else {
                result[0] += -0.057288275878759745;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.06000787050206144;
              } else {
                result[0] += -0.03428915274170302;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += 0.028445476800137648;
                } else {
                  result[0] += -0.06903495925366124;
                }
              } else {
                result[0] += 0.02168906908146647;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += 0.04371348139196859;
              } else {
                result[0] += -0.017564608671433845;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.07197884267887027;
                } else {
                  result[0] += -0.127258598771417;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                  result[0] += -0.01118614757373531;
                } else {
                  result[0] += -0.06683351155747451;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.04756889453621207;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.030973826351920003;
            } else {
              result[0] += -0.07341168824545859;
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += 0.05822875709763242;
                } else {
                  result[0] += -0.03322099447362339;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.913499355316162998) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0900472378924072;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0861185054691414;
                      } else {
                        result[0] += -0.02043835351746832;
                      }
                    } else {
                      result[0] += 0.05047870779114876;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.03852003684504968;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.10207184427904295;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.0795614565394106;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.10134685902295236;
                          } else {
                            result[0] += -0.12782294022626234;
                          }
                        } else {
                          result[0] += 0.10911357820057499;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)14.56755733489990412) ) ) {
                  result[0] += -0.07567227107425473;
                } else {
                  result[0] += 0.13566572662384224;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.08800631927425523;
                } else {
                  result[0] += 0.05223975362393703;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
          result[0] += -0.09444007013850524;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.845905780792238104) ) ) {
            result[0] += -0.06154784367705594;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.06115830343961215;
            } else {
              result[0] += 0.07904860722335111;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.1066109415134054;
        } else {
          result[0] += -0.16139784531547272;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.1337999939902161;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.039454642957172166;
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.02194593066935178;
          } else {
            result[0] += -0.04766582864579632;
          }
        }
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
          result[0] += -0.11754393376318598;
        } else {
          result[0] += -0.0680099246340466;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.06302941735086186;
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.04922700289654272;
              } else {
                result[0] += 0.002906940749908244;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.039407965832109766;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.059720697381514226;
                    } else {
                      result[0] += -0.008543518170962909;
                    }
                  } else {
                    result[0] += -0.08415434226502999;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                  result[0] += 0.02774510516833364;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.08568755608131422;
                  } else {
                    result[0] += 0.004152852476558317;
                  }
                }
              }
            } else {
              result[0] += -0.04228327044567362;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                    result[0] += 0.02287159449970858;
                  } else {
                    result[0] += -0.04585709304024106;
                  }
                } else {
                  result[0] += -0.11434251410595296;
                }
              } else {
                result[0] += -0.11319778736183139;
              }
            } else {
              result[0] += -0.10351025048691183;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.020951951086294718;
                } else {
                  result[0] += 0.09493162162312624;
                }
              } else {
                result[0] += -0.02007765127337594;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.02473107226011611;
                  } else {
                    result[0] += -0.056046559413795105;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.06821166627747809;
                      } else {
                        result[0] += -0.022532271827443995;
                      }
                    } else {
                      result[0] += 0.020807072135922408;
                    }
                  } else {
                    result[0] += 0.029847400595277643;
                  }
                }
              } else {
                result[0] += 0.04053665386678301;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.05759912036918511;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.02140879576365356;
            } else {
              result[0] += -0.08148807406801074;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
                      result[0] += 0.021067727329327927;
                    } else {
                      result[0] += -0.015460282276129785;
                    }
                  } else {
                    result[0] += -0.04863550235639699;
                  }
                } else {
                  result[0] += 0.04618024394002198;
                }
              } else {
                result[0] += 0.08873032597705657;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.05412534909147442;
                    } else {
                      result[0] += -0.017796605895184798;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.06567724669650563;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.07640893436476556;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.13546269986161608;
                          } else {
                            result[0] += 0.07109783104551419;
                          }
                        }
                      } else {
                        result[0] += 0.10218173498789307;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.11189118950403493;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                      result[0] += -0.0601803092897963;
                    } else {
                      result[0] += 0.053442986851711706;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                  result[0] += -0.04727260447028845;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += -0.09087821083527908;
                      } else {
                        result[0] += 0.06449632662756039;
                      }
                    } else {
                      result[0] += 0.07810069504985265;
                    }
                  } else {
                    result[0] += 0.1023142061176841;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          result[0] += -0.1362883912276073;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
            result[0] += -0.11742781330896855;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.10998345903529404;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.022344179075211103;
              } else {
                result[0] += -0.10252754931478342;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          result[0] += -0.06332828733492694;
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.013498428421656586;
          } else {
            result[0] += 0.10313728831025815;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.1227230096646246;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          result[0] += 0.002066901301923294;
        } else {
          result[0] += -0.12357870243260145;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.10845538211384649;
        } else {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
            result[0] += -0.087073109146311;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.020004594430881397;
            } else {
              result[0] += -0.06950550783537483;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.01913670175231762;
                  } else {
                    result[0] += 0.04483177950038278;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                      result[0] += -0.08239284830660715;
                    } else {
                      result[0] += -0.0007492383410163982;
                    }
                  } else {
                    result[0] += 0.06422240816854828;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                  result[0] += 0.03020585896597579;
                } else {
                  result[0] += -0.016520296909689747;
                }
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.08642512989631404;
                  } else {
                    result[0] += -0.007416309833673519;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += -0.036867642563401984;
                  } else {
                    result[0] += 0.009556530966768223;
                  }
                }
              } else {
                result[0] += 0.017292049213214377;
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.10944186740878287;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.009982831866598043;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.03938632220251852;
                  } else {
                    result[0] += 0.06470181569122424;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.499747991561890537) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.011612446568778578;
                } else {
                  result[0] += -0.06752508047032156;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0980827975204191;
                } else {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04817589629106163;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.09055768069908675;
                    } else {
                      result[0] += 0.04139850035799123;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.04351807964575624;
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.11129588519167521;
            } else {
              result[0] += -0.05557379305461718;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.03805600704045854;
            } else {
              result[0] += 0.0015785691960489051;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.021779900353125096;
            } else {
              result[0] += -0.09555510903143677;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.03322131421567972;
            } else {
              result[0] += -0.03045105756866521;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.06250570010157888;
              } else {
                result[0] += -0.030435321831887525;
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.03010331290111595;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.030769350356916844;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12055253982544123) ) ) {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                              result[0] += 0.05965102229928442;
                            } else {
                              result[0] += -0.02131514844772608;
                            }
                          } else {
                            result[0] += 0.05870748506388551;
                          }
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                            result[0] += 0.06812528313506849;
                          } else {
                            result[0] += 0.12388197703221156;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.056081583010231584;
                    }
                  } else {
                    result[0] += 0.08607656553950181;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.08640049348850405;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.07838909885287759;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                      result[0] += 0.057294184421954636;
                    } else {
                      result[0] += -0.08781265918017139;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.1443965394811851;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
              result[0] += -0.11741396301028606;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.99033999443054288) ) ) {
                result[0] += -0.09633731305780706;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.032596201766972054;
                } else {
                  result[0] += -0.0762148114691054;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.0060358072793604;
          } else {
            result[0] += -0.135939216767835;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
          result[0] += -0.09141275374191861;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
            result[0] += -0.06841265564470526;
          } else {
            result[0] += 0.05960958647540007;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.11381124337439391;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
          result[0] += 0.06693773193895346;
        } else {
          result[0] += -0.07869183642027483;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
        result[0] += -0.019202600087229856;
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
          result[0] += -0.10247061309666515;
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.08195412222724091;
          } else {
            result[0] += -0.024143777860686128;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.00041227939896044675;
                  } else {
                    result[0] += 0.0310266358749977;
                  }
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.049975519752124206;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.006380471050040689;
                      } else {
                        result[0] += 0.045151057651054316;
                      }
                    } else {
                      result[0] += -0.04961201319870434;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.274755001068116123) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.04490192393250612;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.520321369171144354) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.015734363363409633;
                        } else {
                          result[0] += 0.06217607840250198;
                        }
                      } else {
                        result[0] += -0.015182970201060176;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                        result[0] += 0.04278426681142765;
                      } else {
                        result[0] += -0.04763364068662947;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.07370503415415564;
                  } else {
                    result[0] += -0.0625645068235771;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.061073623495174756;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.176905632019043857) ) ) {
                    result[0] += -0.013909662568542959;
                  } else {
                    result[0] += 0.01862387758786922;
                  }
                }
              } else {
                result[0] += 0.021614065612570166;
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.13315613691392245;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.046473338456906876;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.001377313810077214;
                    } else {
                      result[0] += -0.09931522160249905;
                    }
                  } else {
                    result[0] += 0.07934130605917526;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                result[0] += -0.005360066383979499;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += 0.03671493879379167;
                  } else {
                    result[0] += -0.08341359023762891;
                  }
                } else {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.012231353435373071;
                    } else {
                      result[0] += -0.0601258706743451;
                    }
                  } else {
                    result[0] += 0.02376792660396006;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.03384136658552428;
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.10235428015626949;
            } else {
              result[0] += -0.051581387030871666;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.04944584221597052;
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.023571881099294027;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.060839427457058975;
                  } else {
                    result[0] += 0.016259894937300932;
                  }
                }
              } else {
                result[0] += -0.0566193314494903;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.0327295251949932;
              } else {
                result[0] += -0.08756167168757112;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.0005893456187507189;
              } else {
                result[0] += 0.03726642773891769;
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.490982532501221591) ) ) {
                    result[0] += 0.06535147095337353;
                  } else {
                    result[0] += 0.008821521259039887;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.03148552730847893;
                    } else {
                      result[0] += 0.05106350180555437;
                    }
                  } else {
                    result[0] += 0.09474007625560761;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.05767400782843819;
                } else {
                  result[0] += 0.04714441290487086;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
        result[0] += -0.10499866029558086;
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.13509082975669712;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
                result[0] += -0.09129578331239838;
              } else {
                result[0] += -0.0047189276749675935;
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                result[0] += 0.013403002362103162;
              } else {
                result[0] += 0.06827213717051081;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += -0.04917819226401614;
          } else {
            result[0] += -0.13639269641238672;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.0572980781109183;
      } else {
        result[0] += -0.02100976529527758;
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.10731267169789988;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            result[0] += -0.03619356674240791;
          } else {
            result[0] += -0.09190128401093063;
          }
        } else {
          result[0] += -0.09701754994157857;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += -0.023881806504903732;
            } else {
              result[0] += -0.0816430466236059;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.687107801437378818) ) ) {
                result[0] += 0.03732689379968528;
              } else {
                result[0] += 0.010530173261376096;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.0330898979914669;
                  } else {
                    result[0] += -0.03089066518952993;
                  }
                } else {
                  result[0] += -0.07532910508740276;
                }
              } else {
                result[0] += -0.07891636160760644;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08811243706288105;
              } else {
                result[0] += 0.04697834825648501;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81517744064331232) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.045470260140445924;
                  } else {
                    result[0] += 0.037862002063196655;
                  }
                } else {
                  result[0] += -0.08158352359758578;
                }
              } else {
                result[0] += -0.10207000072396895;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += -0.027105197850466774;
                  } else {
                    result[0] += 0.027381943251062635;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.27480554580688654) ) ) {
                    result[0] += -0.07845531982670642;
                  } else {
                    result[0] += 0.07904764779775392;
                  }
                }
              } else {
                result[0] += -0.09612179119145124;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.899837255477906162) ) ) {
                  result[0] += -0.008975603734002992;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.06895618818813247;
                  } else {
                    result[0] += 0.014549550446050545;
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.031194752837023245;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.05063140424266097;
                      } else {
                        result[0] += -0.02858490868027115;
                      }
                    } else {
                      result[0] += 0.04597133370695744;
                    }
                  }
                } else {
                  result[0] += -0.027937619239764164;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.00559356818739154;
            } else {
              result[0] += 0.03322625368086101;
            }
          } else {
            result[0] += -0.0717809390355666;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.026048381474534144;
            } else {
              result[0] += -0.03303092391780837;
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.030872345603966973;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += 0.05169837860576637;
                } else {
                  result[0] += -0.03164967040344781;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.004902479626676427;
                        } else {
                          result[0] += -0.09294168442604842;
                        }
                      } else {
                        result[0] += 0.05180738904870698;
                      }
                    } else {
                      result[0] += 0.04460434621445342;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.03508563533164529;
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.04128091412198545;
                          } else {
                            result[0] += -0.08535335543282385;
                          }
                        }
                      } else {
                        result[0] += 0.053599096188701637;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                        result[0] += 0.04188799592417476;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.040991758322595676;
                        } else {
                          result[0] += 0.08384922755466685;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.07767986227757195;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.08034104902688291;
                    } else {
                      result[0] += 0.04267817385847695;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.075335502624512607) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.925687789916993964) ) ) {
            result[0] += -0.09209790143345666;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.08115864889352827;
            } else {
              result[0] += 0.0036386275719576507;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
            result[0] += -0.06695883619347658;
          } else {
            result[0] += 0.05686638779821758;
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.08310466556235112;
          } else {
            result[0] += 0.023981613907496424;
          }
        } else {
          result[0] += -0.11048864305017064;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.040943465939400216;
      } else {
        result[0] += -0.025643860013007492;
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.10037762342468777;
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
          result[0] += -0.09313192034167656;
        } else {
          result[0] += -0.05166932936976101;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                result[0] += 0.01042803851532097;
              } else {
                result[0] += -0.03813679981348972;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                  result[0] += 0.0185431325440403;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += 0.015637074091262415;
                  } else {
                    result[0] += -0.04505768043849312;
                  }
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.003213401655086118;
                    } else {
                      result[0] += 0.0377119541073627;
                    }
                  } else {
                    result[0] += 0.041535293624548705;
                  }
                } else {
                  result[0] += -0.04640405681473165;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.01500539399680225;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += -0.06835359188868137;
                  } else {
                    result[0] += 0.0066895928633348535;
                  }
                }
              } else {
                result[0] += -0.07150860255487043;
              }
            } else {
              result[0] += -0.06860834067186616;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.0992652684097999;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.0011366075405204869;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += -0.019993279189321755;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.06868398600900713;
                  } else {
                    result[0] += 0.02421738661279751;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.013218283982741894;
                } else {
                  result[0] += -0.062916772749977;
                }
              } else {
                result[0] += -0.055887195895418844;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.09049488903713744;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.5708808898925799) ) ) {
                    result[0] += 0.012758922107601867;
                  } else {
                    result[0] += -0.07368047996122824;
                  }
                }
              } else {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.05559608192756965;
                  } else {
                    result[0] += -0.019602520312073053;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.07547454607192183;
                  } else {
                    result[0] += 0.031901999024211064;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.00453639393386159;
              } else {
                result[0] += 0.053517070301299687;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                result[0] += -0.047526182379440736;
              } else {
                result[0] += 0.02924858127058192;
              }
            }
          } else {
            result[0] += -0.06565535894678962;
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.03359334068925587;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.016511708551684635;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.01478007980558075;
                      } else {
                        result[0] += 0.038182330429756765;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += 0.038576107674811705;
                        } else {
                          result[0] += 0.0959037136786464;
                        }
                      } else {
                        result[0] += 0.10243310846830528;
                      }
                    }
                  } else {
                    result[0] += -0.05203909730133385;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.0866207059716621;
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.047292859205578776;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.12914980299120546;
                      } else {
                        result[0] += 0.04608688676377837;
                      }
                    }
                  } else {
                    result[0] += 0.08668547699362139;
                  }
                }
              }
            } else {
              if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.05350971703950628;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.002798297080949439;
                } else {
                  result[0] += 0.05197211871141119;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.075335502624512607) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
            result[0] += -0.08925199210047603;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += -0.08739634079977122;
            } else {
              result[0] += -0.00819878504357565;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.668367385864259589) ) ) {
            result[0] += -0.06486094595128024;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.03671181108279225;
            } else {
              result[0] += 0.05758398723136037;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.03674497371008453;
        } else {
          result[0] += -0.10462677100608249;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
        result[0] += 0.005793861312189264;
      } else {
        result[0] += -0.12204376836579695;
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.09444567364749726;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.025735156491258373;
          } else {
            result[0] += -0.07476642694389243;
          }
        } else {
          result[0] += -0.08609957259006774;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.02049149641347142;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.687107801437378818) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.014688771693176337;
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                      result[0] += 0.028146145686385085;
                    } else {
                      result[0] += 0.05733563239014882;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.00024186629326029545;
                    } else {
                      result[0] += 0.04096230054171182;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                  result[0] += 0.01978540401061228;
                } else {
                  result[0] += -0.021805575288594177;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += -0.005280242350097374;
              } else {
                result[0] += -0.07011112755770897;
              }
            } else {
              result[0] += -0.06591180616955061;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.0936362741850723;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += 0.02921078589218334;
                } else {
                  result[0] += -0.033808768249226545;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.725620865821838823) ) ) {
                  result[0] += -0.04769574312162586;
                } else {
                  result[0] += 0.025406854543288;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.0401527823853851;
                } else {
                  result[0] += -0.04775868543234342;
                }
              } else {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10940361022949396) ) ) {
                    result[0] += 0.007812570302743424;
                  } else {
                    result[0] += -0.03348135566725884;
                  }
                } else {
                  result[0] += 0.031117319153435814;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.08914692297943967;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35526132583618342) ) ) {
                    result[0] += 0.009318558601532663;
                  } else {
                    result[0] += -0.07569583067489549;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.07002822948906026;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.59645986557007014) ) ) {
                      result[0] += 0.05864832694014654;
                    } else {
                      result[0] += -0.05113536438956763;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06714868096942989;
                    } else {
                      result[0] += -0.00747484740604552;
                    }
                  } else {
                    result[0] += 0.01968423286777896;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += 0.011112517435021695;
              } else {
                result[0] += -0.03604274855390731;
              }
            } else {
              result[0] += 0.03465732255808703;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.011392070934348755;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
                result[0] += -0.08406287722465644;
              } else {
                result[0] += -0.014974144995930694;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.016211150943099845;
            } else {
              result[0] += -0.02923200637451187;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.04800058757676993;
              } else {
                result[0] += -0.0354146452834618;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.05223986296135574;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.049099057575791916;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                      result[0] += -0.017597082055926545;
                    } else {
                      result[0] += 0.03777697134302613;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.01600084936140461;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0033863145923916623;
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.07410188673222416;
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                            result[0] += -0.056484617973011964;
                          } else {
                            result[0] += 0.07008844281323952;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.07318017684089961;
                    }
                  } else {
                    result[0] += -0.026250573392317535;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.075335502624512607) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
            result[0] += -0.07944819254272198;
          } else {
            result[0] += -0.021840760483496924;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.668367385864259589) ) ) {
            result[0] += -0.061985052717666925;
          } else {
            result[0] += 0.04521475960250229;
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.03448426253840417;
        } else {
          result[0] += -0.09936795688413921;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.018846329796582566;
          } else {
            result[0] += 0.07039831237764889;
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.018187158045951963;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.007004244095705287;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.07594631031844642;
                } else {
                  result[0] += -0.01292901529646208;
                }
              } else {
                result[0] += -0.10731895098265842;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.08875739854946693;
        } else {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
            result[0] += -0.08710400344811313;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.066924138271933;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.03819270753573878;
                } else {
                  result[0] += -0.061662494958318284;
                }
              } else {
                result[0] += -0.07471309976953422;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.05900373672965758;
      } else {
        result[0] += 0.014708510867268796;
      }
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
              result[0] += -0.0038233228182875468;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.668153762817383701) ) ) {
                result[0] += -0.026716296769360222;
              } else {
                result[0] += -0.07285493015508607;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
              result[0] += -0.02967469208085971;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.027007986950946434;
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                  result[0] += 0.013962183143163005;
                } else {
                  result[0] += -0.08602414017936981;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.03660739948619112;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.07828024513980544;
                  } else {
                    result[0] += 0.001493704578049314;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                  result[0] += 0.0171147983914065;
                } else {
                  result[0] += -0.04919225157143108;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.07634918261020916;
                } else {
                  result[0] += -0.008263028778390871;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.004605849878158596;
                  } else {
                    result[0] += 0.06954257849484928;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                    result[0] += 0.011539810558986107;
                  } else {
                    result[0] += 0.05813494211974823;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += -0.021858012901881933;
            } else {
              result[0] += 0.009297637982750311;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
                    result[0] += 0.026745783340051968;
                  } else {
                    result[0] += 0.04706437844953084;
                  }
                } else {
                  result[0] += 0.005150358480874312;
                }
              } else {
                result[0] += -0.06147509289289049;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.02969435966419595;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.07562805416127816;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.06542903919979215;
                  } else {
                    result[0] += 0.0032391963595022593;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.02378261098469265;
              } else {
                result[0] += -0.04376821980714535;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.01636200716062807;
              } else {
                result[0] += 0.0499484092902486;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                result[0] += -0.0003918533930742521;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.308072090148926669) ) ) {
                  result[0] += -0.030542320066298496;
                } else {
                  result[0] += -0.08224110023051902;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                result[0] += -0.03427959546651314;
              } else {
                result[0] += 0.010927453802629746;
              }
            }
          } else {
            result[0] += 0.016214607812882458;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.615975379943848544) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
            result[0] += -0.08620007723858508;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.04900604911743303;
            } else {
              result[0] += 0.008044476188107713;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.1735671379777394;
            } else {
              result[0] += 0.009634568670579562;
            }
          } else {
            result[0] += 0.054374191935993826;
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.05761890924258631;
          } else {
            result[0] += -0.10483512020436273;
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.0660730577864218;
          } else {
            result[0] += 0.013584524566566895;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
        result[0] += -0.000403792444873411;
      } else {
        result[0] += -0.11144839108286331;
      }
    } else {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.08402941947160027;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.042666470556027236;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.10716590339156958;
              } else {
                result[0] += -0.03136387591191286;
              }
            } else {
              result[0] += -0.0937157605439045;
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.006279761472939001;
            } else {
              result[0] += -0.0851968558225201;
            }
          } else {
            result[0] += -0.07472289516286017;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.019462023291329327;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                result[0] += 0.027707122657749324;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                  result[0] += 0.01825085193983735;
                } else {
                  result[0] += -0.022710500731387008;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += -0.006061790418638396;
              } else {
                result[0] += -0.06616368362428574;
              }
            } else {
              result[0] += -0.06266783895340877;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.08549160109385931;
            } else {
              result[0] += 0.010495232023527687;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.0159884905479959;
              } else {
                result[0] += -0.06660808030380437;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.08116447517225539;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.66412305831909357) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.897119760513306552) ) ) {
                        result[0] += 0.06008328939427528;
                      } else {
                        result[0] += -0.05270421331425306;
                      }
                    } else {
                      result[0] += -0.06807131066799563;
                    }
                  }
                } else {
                  result[0] += 0.040328062380468016;
                }
              } else {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.0450569504973025;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
                      result[0] += 0.024348151544132193;
                    } else {
                      result[0] += -0.02701515111478698;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.056815695683834544;
                  } else {
                    result[0] += 0.022051632735332963;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.04517407345941199;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.009981396141622715;
                } else {
                  result[0] += -0.03605614812923512;
                }
              } else {
                result[0] += 0.03482060268049106;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.001828372996139064;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.274755001068116123) ) ) {
                  result[0] += -0.07498658373450766;
                } else {
                  result[0] += -0.01081580041681109;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.030201279693870122;
                  } else {
                    result[0] += 0.007515891076484102;
                  }
                } else {
                  result[0] += 0.028151385284636277;
                }
              } else {
                result[0] += 0.04952608275535561;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.04923550659970538;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.050453573826283904;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                      result[0] += -0.02067933863418546;
                    } else {
                      result[0] += 0.04126195756680985;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                    result[0] += -0.02990237588571701;
                  } else {
                    result[0] += 0.02872062607006387;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.051305146547530545;
                    } else {
                      result[0] += -0.02380160984666925;
                    }
                  } else {
                    result[0] += 0.07423245304944605;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.10412361006257874;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
              result[0] += -0.08642821478040245;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.10823514753663457;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                  result[0] += -0.09329248618752732;
                } else {
                  result[0] += 0.011228194360081736;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.025703854202291793;
          } else {
            result[0] += -0.11186342289333631;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.561026811599732333) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.149475097656251776) ) ) {
            result[0] += -0.0781974749004413;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += -0.0664923535691002;
            } else {
              result[0] += 0.009408303764878966;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
            result[0] += -0.05122149565394296;
          } else {
            result[0] += 0.05012752112023303;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.02127287126789469;
        } else {
          result[0] += -0.06597911787587603;
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.05851804322223567;
            } else {
              result[0] += -0.010331823096383001;
            }
          } else {
            result[0] += -0.09005124833392703;
          }
        } else {
          result[0] += -0.04706918587014647;
        }
      }
    } else {
      if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.0784189250406641;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.0423096435306694;
        } else {
          result[0] += -0.08340064126125196;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                result[0] += -0.035215133929630896;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.050340953821774295;
                  } else {
                    result[0] += 0.0019299179891114834;
                  }
                } else {
                  result[0] += 0.022043792587671023;
                }
              }
            } else {
              result[0] += -0.02412985651150112;
            }
          } else {
            result[0] += -0.08964378423569734;
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0700503225801712;
            } else {
              result[0] += 0.023774188441106608;
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += 0.019635063781926337;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                  result[0] += -0.013172466406286954;
                } else {
                  result[0] += -0.06241088398395364;
                }
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.019042811062069443;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.04955619858956196;
                    } else {
                      result[0] += 0.007746927889647004;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                    result[0] += -0.023481376188156374;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.009015212182811424;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.0772868075285498;
                          } else {
                            result[0] += -0.1145323733423507;
                          }
                        } else {
                          result[0] += 0.03625983961887872;
                        }
                      } else {
                        result[0] += 0.06877414730217477;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.01592340792747471;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.448888063430787021) ) ) {
                    result[0] += -0.03634744310838628;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                      result[0] += 0.005082263302937182;
                    } else {
                      result[0] += 0.15239044073132155;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0002118602981867802;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                result[0] += 0.018111215001550694;
              } else {
                result[0] += -0.02522358976248311;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.021017082219958957;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.031898467674923374;
                } else {
                  result[0] += -0.0257951199153932;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                result[0] += -0.00015446526322226254;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                  result[0] += -0.014084418335693218;
                } else {
                  result[0] += -0.05837502469514242;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                result[0] += -0.030213097353567805;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.026949204694148333;
                } else {
                  result[0] += 0.010963357527467237;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
                result[0] += 0.012731930529264924;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                  result[0] += -0.0026246192845553948;
                } else {
                  result[0] += -0.0574417195936503;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.022331401498122693;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.02636511513087962;
                    } else {
                      result[0] += -0.043243975207837376;
                    }
                  } else {
                    result[0] += -0.029851672081189426;
                  }
                }
              } else {
                result[0] += 0.0451298801661741;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.101423740386963779) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12055253982544123) ) ) {
            result[0] += -0.065383308846541;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
              result[0] += -0.08653744682939296;
            } else {
              result[0] += -0.0017051252871631275;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
            result[0] += -0.05893344103428016;
          } else {
            result[0] += 0.036947852083973796;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.11196022216430884;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
              result[0] += -0.08213010701863468;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                result[0] += -0.07934327522532059;
              } else {
                result[0] += 0.018474704945739134;
              }
            }
          }
        } else {
          result[0] += -0.1090801106052572;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.03970658258617207;
      } else {
        result[0] += -0.021208326124415748;
      }
    } else {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.07394746119380828;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
            result[0] += 0.007447587551531145;
          } else {
            result[0] += -0.050150029758257386;
          }
        } else {
          result[0] += -0.07915735601501832;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0039616511352722735;
                    } else {
                      result[0] += -0.05581281965846006;
                    }
                  } else {
                    result[0] += -0.0336409733672621;
                  }
                } else {
                  result[0] += -0.07017332363125471;
                }
              } else {
                result[0] += 0.008517626327082903;
              }
            } else {
              result[0] += -0.08487415564593617;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
              result[0] += -0.023889995601196982;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.021916242019236067;
              } else {
                result[0] += 0.012241867346900377;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06228919481620078;
            } else {
              result[0] += 0.021642916151603433;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0911903381347674) ) ) {
                result[0] += 0.008476370189938758;
              } else {
                result[0] += -0.04618872855621408;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.013112537826315128;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.07289822514133827;
                    } else {
                      result[0] += 0.021488001582051927;
                    }
                  }
                } else {
                  result[0] += -0.0022953052058231733;
                }
              } else {
                result[0] += -0.025589529154033154;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.568724632263184482) ) ) {
                  result[0] += 0.0202294140602359;
                } else {
                  result[0] += 0.03797298710448666;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.03607921682766665;
                  } else {
                    result[0] += -0.03975887673094163;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
                    result[0] += 0.008749347212005987;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.08281049282715192;
                    } else {
                      result[0] += 0.00311320391483798;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.002111527533492349;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += 0.02208503156791422;
              } else {
                result[0] += -0.026294665546465923;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.010870751079895042;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.036054282112915545;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                      result[0] += 0.03627067522193788;
                    } else {
                      result[0] += 0.08648455040110561;
                    }
                  } else {
                    result[0] += -0.04249359172754587;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                result[0] += 0.0014736463761668241;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                  result[0] += -0.013753325962750926;
                } else {
                  result[0] += -0.05707432228212278;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                result[0] += -0.042326834391265694;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.019906192545375682;
                } else {
                  result[0] += 0.009295619532414834;
                }
              }
            }
          } else {
            result[0] += 0.011093096842134475;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.561026811599732333) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.196324348449708808) ) ) {
            result[0] += -0.06830882462154479;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.06108894562456797;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += 0.05248615254774759;
              } else {
                result[0] += -0.008641699907545475;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.18962340447220888;
              } else {
                result[0] += 0.003831569924245687;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                result[0] += -0.06607832245881644;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.1020140338993415;
                } else {
                  result[0] += 0.06619775173987359;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.025365080557291343;
            } else {
              result[0] += 0.09844087195788148;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            result[0] += -0.05980246133695079;
          } else {
            result[0] += 0.029561052899258086;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.547126770019532138) ) ) {
            result[0] += -0.09885787097303766;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.07598158956490386;
              } else {
                result[0] += 0.06338394193871966;
              }
            } else {
              result[0] += -0.08729653228879924;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.057556199003123656;
      } else {
        result[0] += -0.009757074203229918;
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.0714665599035851;
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += -0.03528903798464686;
              } else {
                result[0] += -0.12484069816109218;
              }
            } else {
              result[0] += -0.07587017215565837;
            }
          } else {
            result[0] += -0.08989574930836985;
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.049646978532719974;
          } else {
            result[0] += -0.008100972422838854;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              result[0] += -0.004941834401551447;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                result[0] += -0.025113677371274776;
              } else {
                result[0] += -0.0638128382575968;
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.028538276453568896;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.09134718656297341;
              } else {
                result[0] += -0.005270641814311072;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
            result[0] += -0.030035629714795;
          } else {
            result[0] += 0.0023485448086782374;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.09745602338541087;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.006226730063585838;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                result[0] += -0.025317903728229386;
              } else {
                result[0] += 0.05182289791545616;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.07263931703127545;
            } else {
              result[0] += -0.015570406457609957;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)14.32165384292602717) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.01726841350884404;
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.05091717635429286;
                } else {
                  result[0] += 0.0034377089808085032;
                }
              }
            } else {
              result[0] += 0.13907193033570472;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.016325364362613646;
              } else {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06627038352480123;
                } else {
                  result[0] += 0.013069655565713835;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.013283613194768671;
              } else {
                result[0] += 0.020848104765194293;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                  result[0] += 0.015207948021049515;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                    result[0] += 0.010943732656213696;
                  } else {
                    result[0] += -0.050937854123364656;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
                    result[0] += 0.035360611756779674;
                  } else {
                    result[0] += -0.05780025666105443;
                  }
                } else {
                  result[0] += 0.02515720182742085;
                }
              }
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06166232889047969;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.009206561560729853;
                } else {
                  result[0] += 0.07397403568034891;
                }
              }
            }
          }
        } else {
          result[0] += -0.08751018719182556;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.006549638983362561;
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0911903381347674) ) ) {
                  result[0] += -0.017183687883100117;
                } else {
                  result[0] += -0.056646868028751754;
                }
              } else {
                result[0] += 0.019850122942367883;
              }
            } else {
              result[0] += -0.08266529937020163;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
              result[0] += -0.07309621553672786;
            } else {
              result[0] += 0.01172052938041961;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                  result[0] += 0.021067132522937432;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += 0.011693164614749242;
                  } else {
                    result[0] += -0.06318269118297785;
                  }
                }
              } else {
                result[0] += 0.023819606975712816;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10166215896606623) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.027529172881210786;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.08037925305366082;
                        } else {
                          result[0] += 0.015043441497165006;
                        }
                      }
                    } else {
                      result[0] += -0.019842632969134216;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.06840684956020195;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                        result[0] += -0.07242505314990451;
                      } else {
                        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.012276324867993568;
                        } else {
                          result[0] += 0.07277682372405059;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.05702280637464126;
                }
              } else {
                result[0] += -0.04919014336458996;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.02855717688407991;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.014713708910549288;
              } else {
                result[0] += -0.054413067951918116;
              }
            } else {
              result[0] += 0.00015505814863498195;
            }
          } else {
            result[0] += -0.09273349387282953;
          }
        }
      } else {
        result[0] += -0.09338413684697401;
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.06894665187163275;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.06878455019351733;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.01001338248558273;
            } else {
              result[0] += -0.06955608810531537;
            }
          }
        } else {
          result[0] += -0.08870044740021199;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.0008751506452592903;
              } else {
                result[0] += -0.055209366939342946;
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.04742598156151919;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.04562333243064246;
                    } else {
                      result[0] += 0.020989800154527435;
                    }
                  }
                } else {
                  result[0] += -0.07724190623586669;
                }
              } else {
                result[0] += 0.011033688503391878;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
              result[0] += -0.028154020937886732;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.015254777138965617;
              } else {
                result[0] += 0.010434362245394381;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0911903381347674) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.002489859610404906;
                } else {
                  result[0] += 0.033181422177360316;
                }
              } else {
                result[0] += -0.010265402152334136;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.057857036942800005;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.02135397148063529;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.10968606716803762;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.0010674409371683915;
                    } else {
                      result[0] += 0.06997346681252611;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.07507304285463873;
              } else {
                result[0] += -0.009106915061740184;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                  result[0] += 0.022640908337411232;
                } else {
                  result[0] += -0.04714297049363111;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.040529173045197354;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.05813601456093642;
                    } else {
                      result[0] += 0.06854467728739702;
                    }
                  } else {
                    result[0] += 0.03529008652945499;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0014605552812293138;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.018498685440033285;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.050561014547241656;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
                    result[0] += -0.06394935824281646;
                  } else {
                    result[0] += 0.053228875546832255;
                  }
                }
              }
            } else {
              result[0] += 0.03287251526996206;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.005634472583965827;
              } else {
                result[0] += -0.02195118118102875;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                result[0] += -0.04024832486636001;
              } else {
                result[0] += 0.016082135775646177;
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += 0.0018290360554533683;
              } else {
                result[0] += -0.04616951871889974;
              }
            } else {
              result[0] += 0.012886308385582393;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          result[0] += -0.09544612945743136;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
            result[0] += -0.08077697014393204;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.07032991355483663;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                  result[0] += -0.011328497039256643;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.012640058851597852;
                  } else {
                    result[0] += 0.07549216736112037;
                  }
                }
              } else {
                result[0] += -0.0555313680709723;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.01497408059301299;
          } else {
            result[0] += -0.07738584911905039;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.99033999443054288) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.059382231343279517;
            } else {
              result[0] += 0.035004502228258315;
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.04788013166489477;
            } else {
              result[0] += 0.10335425182415706;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
          result[0] += -0.007333310178658733;
        } else {
          result[0] += -0.07494431062254059;
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.06262281337124619;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += -0.03615935552988952;
          } else {
            result[0] += -0.07264855088226004;
          }
        }
      }
    } else {
      result[0] += 0.008602418687502576;
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.03239376121406319;
          } else {
            result[0] += -0.0038191328302122907;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
            result[0] += -0.0010771896498116221;
          } else {
            result[0] += -0.06460838527929023;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.03608890622175106;
                    } else {
                      result[0] += 0.06884116028062405;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                      result[0] += 0.03389161370571758;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0077874502532165775;
                      } else {
                        result[0] += -0.057039406399610505;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.008346154903898863;
                    } else {
                      result[0] += -0.033325792385726645;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                      result[0] += -0.023341926614317297;
                    } else {
                      result[0] += 0.034457511200411704;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += 0.016410756820619512;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += -0.04872369154229492;
                    } else {
                      result[0] += 0.013308996694981166;
                    }
                  }
                } else {
                  result[0] += 0.026459937781148343;
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0358300512189615;
                  } else {
                    result[0] += -0.0799408252121838;
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.017607616584036326;
                  } else {
                    result[0] += -0.05347613253604369;
                  }
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += -0.025565038778951367;
                  } else {
                    result[0] += 0.02786090278356725;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                    result[0] += -0.04007727127718235;
                  } else {
                    result[0] += 0.006194034976964628;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.010391477999588436;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.594409704208374912) ) ) {
                  result[0] += -0.06338796735337458;
                } else {
                  result[0] += 0.019645087996907146;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04777658356293324;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04976163103995958;
                  } else {
                    result[0] += 0.019746144855586692;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.343781709671021396) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.196324348449708808) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.005733301525555137;
                    } else {
                      result[0] += -0.08271113035050456;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.06671416572273234;
                      } else {
                        result[0] += -0.047064959589649236;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        result[0] += 0.04574073618944621;
                      } else {
                        result[0] += -0.03356333408229429;
                      }
                    }
                  }
                } else {
                  result[0] += 0.06689806011168332;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.049233715003856274;
                    } else {
                      result[0] += 0.06832638051282366;
                    }
                  } else {
                    result[0] += -0.0010057848285466558;
                  }
                } else {
                  result[0] += -0.045369192673125947;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.02154505307866398;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
                    result[0] += -0.011177273254898545;
                  } else {
                    result[0] += -0.06820831758950012;
                  }
                }
              }
            } else {
              result[0] += 0.04096687702094509;
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.10385292040802946;
                } else {
                  result[0] += -0.03254921029058593;
                }
              } else {
                result[0] += -0.007589299327094407;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                result[0] += -0.04625293674672786;
              } else {
                result[0] += 0.04084194861171529;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.041619474071552844;
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += -0.1123893364160486;
          } else {
            result[0] += -0.037301243040267246;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.561026811599732333) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
            result[0] += -0.07291809152494137;
          } else {
            result[0] += -0.0087690834036872;
          }
        } else {
          result[0] += 0.040727994211492334;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
          result[0] += -0.008271792284562101;
        } else {
          result[0] += -0.06670667756490319;
        }
      } else {
        if ( UNLIKELY(  (data[31].missing != -1) && (data[31].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.05874207900689789;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07380901932010075;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.05267242270945199;
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.032390780034249236;
                  } else {
                    result[0] += -0.03202518984575955;
                  }
                } else {
                  result[0] += -0.06358007615073713;
                }
              }
            }
          } else {
            result[0] += -0.07808287863303595;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.777633190155030185) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.032315270140470025;
        } else {
          result[0] += -0.023982161561855758;
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.0694939100836191;
        } else {
          result[0] += -0.00132573192184649;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.02977978546406872;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.453179836273194248) ) ) {
                  result[0] += -0.010046326031916299;
                } else {
                  result[0] += -0.07739851688768828;
                }
              } else {
                result[0] += 0.023352318181666807;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                  result[0] += 0.018896884877288203;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                    result[0] += -0.021953883652080584;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.06803835596158625;
                    } else {
                      result[0] += -0.09613452410642145;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12769031524658381) ) ) {
                  result[0] += -0.023987999823424797;
                } else {
                  result[0] += -0.08925246806564868;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
            result[0] += -4.254660855345747e-05;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.08518620936970805;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
                result[0] += 0.00735342378623374;
              } else {
                result[0] += -0.061897453810885905;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.00406288342220656;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05779230674288543;
                } else {
                  result[0] += -0.011987339526695255;
                }
              }
            } else {
              result[0] += 0.003902558718650653;
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.012817662520245368;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.011095684397404187;
              } else {
                result[0] += -0.023982745513945483;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.802100181579590732) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.025994601088019372;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.008023390837714172;
                } else {
                  result[0] += -0.017782160221359387;
                }
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.013904069183235918;
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.032943745225394276;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.04565135180842902;
                    } else {
                      result[0] += -0.04903796053413006;
                    }
                  } else {
                    result[0] += -0.015376565709305696;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.00490984139691909;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += 0.05816516047337816;
                  } else {
                    result[0] += -0.014112988268335064;
                  }
                } else {
                  result[0] += -0.0310142826057324;
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.017438061046419335;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.017773385451612164;
                    } else {
                      result[0] += -0.029772846524414306;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                      result[0] += 0.0014981061770648586;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.0030485719243590376;
                      } else {
                        result[0] += 0.04789291609297896;
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
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += -0.13450754353102068;
            } else {
              result[0] += -0.03320527371461259;
            }
          } else {
            result[0] += -0.021917927229211154;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            result[0] += -0.05455402084293609;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.03328231601241755;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0330474913048456;
                } else {
                  result[0] += 0.025527552261902217;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0008020740909364275;
                } else {
                  result[0] += 0.07257727653395221;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.05228286812357272;
        } else {
          result[0] += -0.10950530763893365;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
      result[0] += -0.0012942419822664457;
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.05815102587837966;
        } else {
          result[0] += -0.0252785735561451;
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.07758878255533574;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += -0.02785075808992843;
            } else {
              result[0] += -0.11158060902034697;
            }
          }
        } else {
          result[0] += -0.10630701721728256;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                      result[0] += 0.020951819120333774;
                    } else {
                      result[0] += -0.012885775109560306;
                    }
                  } else {
                    result[0] += 0.028422490883329418;
                  }
                } else {
                  result[0] += -0.03375205783295922;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.059839507156689814;
                } else {
                  result[0] += -0.012206683095768537;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                result[0] += 0.018323137702710407;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0117464155092133;
                  } else {
                    result[0] += -0.0694855376520213;
                  }
                } else {
                  result[0] += 0.045028444149944016;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.006739130134818173;
              } else {
                result[0] += -0.07800296913552751;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  result[0] += 0.01691988202262108;
                } else {
                  result[0] += -0.07936662281621261;
                }
              } else {
                result[0] += 0.03657357250637715;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.0015929444160171513;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.026069699434876237;
                  } else {
                    result[0] += -0.07490680043775019;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.06267152641637057;
                  } else {
                    result[0] += -0.045097785127396706;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += -0.09063438213703692;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                    result[0] += -0.10021222089791584;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.04199264076577526;
                    } else {
                      result[0] += -0.05664489220735811;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.027933800675809944;
              } else {
                result[0] += -0.032970582207519604;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.0012321522169062812;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0028527938412177527;
            } else {
              result[0] += -0.07586333167059756;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03152877415468334;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += 0.011753713359542006;
                } else {
                  result[0] += -0.030456541500986475;
                }
              }
            } else {
              result[0] += -0.04702673613059158;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.03741524226768513;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.007175098077229907;
                } else {
                  result[0] += 0.051515213396505055;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.035840353384914;
                    } else {
                      result[0] += -0.015013824329800394;
                    }
                  } else {
                    result[0] += -0.035140121642937026;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.02605704728251056;
                  } else {
                    if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.08619051408145406;
                    } else {
                      result[0] += -0.0362389165682284;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                  result[0] += -0.049167950620448035;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.013562630585895989;
                  } else {
                    if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.03779463697884408;
                      } else {
                        result[0] += 0.08478871360749207;
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.524927973747253862) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.05062293473817684;
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.1742983971222437;
                              } else {
                                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                                  result[0] += -0.17035760956082452;
                                } else {
                                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                    result[0] += -0.08343614022011832;
                                  } else {
                                    result[0] += 0.06834828567224262;
                                  }
                                }
                              }
                            } else {
                              result[0] += 0.020091153397575973;
                            }
                          } else {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                              result[0] += 0.048270705098828234;
                            } else {
                              result[0] += -0.06548674425810194;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.05608646775470253;
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
      result[0] += -0.05390735554103287;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.026393738400847473;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          result[0] += -0.00873881567382776;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.06980424009827095;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                result[0] += 0.018156771824860442;
              } else {
                result[0] += -0.03717050223036551;
              }
            } else {
              result[0] += -0.10697611243647132;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.053976699231228324;
        } else {
          result[0] += -0.03379072060499186;
        }
      } else {
        result[0] += -0.0711572734381383;
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              result[0] += -0.002955592124661668;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.768316030502320224) ) ) {
                result[0] += -0.019085467884185497;
              } else {
                result[0] += -0.054822809726272716;
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.02282264432775559;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.07687427362265786;
              } else {
                result[0] += -0.004786488413719589;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.0028687769636422545;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.378218650817871982) ) ) {
                result[0] += -0.015633271327752146;
              } else {
                result[0] += -0.06118760124294684;
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.0315518527404711;
              } else {
                result[0] += 0.010034841182547166;
              }
            } else {
              result[0] += -0.02227836363535267;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
          result[0] += 0.0035619979305067935;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
            result[0] += -0.06645797779493427;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.005468206933206313;
            } else {
              result[0] += -0.07954540755138087;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += -0.09255222423229453;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.04533865698301617;
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.07826217017089107;
                } else {
                  result[0] += 0.015518321155257454;
                }
              }
            }
          } else {
            result[0] += 0.0005900064967800408;
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
                    result[0] += 0.01261205308938547;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.04973769189270699;
                      } else {
                        result[0] += 0.0008075174408352962;
                      }
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.06793961144308844;
                      } else {
                        result[0] += 0.03283990142084248;
                      }
                    }
                  }
                } else {
                  result[0] += 0.0017715737802012001;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.011981929448736196;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                      result[0] += -0.020599293440354178;
                    } else {
                      result[0] += -0.08454346037770677;
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.06361636702196129;
                      } else {
                        result[0] += 0.02629585489988195;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.003403709831676393;
                        } else {
                          result[0] += -0.06948147103059145;
                        }
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.018029965930946905;
                        } else {
                          result[0] += 0.04723146620300845;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += -0.002064533164367039;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.027994278430819076;
                    } else {
                      result[0] += 0.05413730435961576;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.469231128692627841) ) ) {
                    result[0] += -0.009161688529460778;
                  } else {
                    result[0] += -0.06546457480005088;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.04093658354707716;
                  } else {
                    result[0] += -0.037985896119035474;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.176905632019043857) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += 0.023288894480302268;
                    } else {
                      result[0] += -0.030753616612025726;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.005280334816582556;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.013489892497453816;
                      } else {
                        result[0] += 0.06202161697578588;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.07546706194035095;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              result[0] += 0.009097709675265487;
            } else {
              result[0] += -0.03022476556515183;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
              result[0] += -0.030717288695088902;
            } else {
              result[0] += 0.0012871244561735828;
            }
          }
        } else {
          result[0] += 0.007517669100953564;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.04737533283711524;
      } else {
        result[0] += -0.01060702623232935;
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.052482566475700744;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.0065475623696208745;
            } else {
              result[0] += -0.045357957989925396;
            }
          } else {
            result[0] += -0.04679164385783361;
          }
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            result[0] += -0.0425732150602231;
          } else {
            result[0] += -0.07220275680031926;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.262283086776734287) ) ) {
            result[0] += -0.10658464440153806;
          } else {
            result[0] += 0.14909966634915603;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.109245061874390537) ) ) {
              result[0] += 0.01178453469371003;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                result[0] += -0.029954691560262876;
              } else {
                result[0] += 0.0010835160158783053;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              result[0] += -0.016204908279882583;
            } else {
              result[0] += -0.06113429871238042;
            }
          }
        } else {
          result[0] += -0.06160251185447294;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.08764856577371097;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10166215896606623) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.025027122647687784;
                  } else {
                    result[0] += -0.0006185712619309903;
                  }
                } else {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.010008831633579409;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                      result[0] += -0.013745246836227524;
                    } else {
                      result[0] += -0.06004746012671931;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.02896586220462983;
                  } else {
                    result[0] += -0.03293283802317865;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                    result[0] += -0.07965337942266737;
                  } else {
                    result[0] += -0.021278992041916005;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                result[0] += -0.03580426082085165;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0332169649115325;
                      } else {
                        result[0] += -0.05145865660627518;
                      }
                    } else {
                      result[0] += 0.016818874752395355;
                    }
                  } else {
                    result[0] += 0.04968145282732388;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.04980498139082277;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.303973913192749912) ) ) {
                          result[0] += -0.11990199382840117;
                        } else {
                          result[0] += -0.0044594857342622135;
                        }
                      } else {
                        result[0] += 0.01255334532942523;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.030218463090281852;
                    } else {
                      result[0] += 0.06060623459200786;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.008368699462762237;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.10334451441758177;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.11230719292934246;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.0883984565734881) ) ) {
                      result[0] += -0.056413344246671826;
                    } else {
                      result[0] += 0.11593127425554167;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.775935173034668857) ) ) {
                    result[0] += -0.04130799295398461;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.12126067697221339;
                    } else {
                      result[0] += -0.006779218789251225;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.05361016547645098;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
                  result[0] += -0.030846769707598483;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
                    result[0] += -0.004057600802235561;
                  } else {
                    result[0] += 0.05936880080711381;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                result[0] += -0.017652899886305258;
              } else {
                result[0] += -0.09376563949755733;
              }
            } else {
              result[0] += 0.00015697260308775824;
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
                result[0] += -0.010820436790505074;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.013693935430119387;
                } else {
                  result[0] += 0.03517201122339004;
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += 0.01366236180926864;
              } else {
                result[0] += 0.02580322021402123;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
              result[0] += -0.04945490246624938;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.99033999443054288) ) ) {
                result[0] += -0.01734789832244859;
              } else {
                result[0] += 0.05715809737624277;
              }
            }
          } else {
            result[0] += -0.06235038070844552;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.010376465504465167;
        } else {
          result[0] += 0.006991459958313514;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
        result[0] += -0.0058331600421786135;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.00853693719512874;
          } else {
            result[0] += -0.05177916881097936;
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.07089072272352358;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.003980820714500734;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.050097413976269545;
              } else {
                result[0] += -0.05679443751325175;
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.007213982737870647;
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += 0.025687609406698342;
          } else {
            result[0] += 0.0015062725788528074;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.03998929843937274;
            } else {
              result[0] += 0.009436258777332287;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.07483157314048484;
              } else {
                result[0] += -0.020020017120326444;
              }
            } else {
              result[0] += -0.018205375046357913;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.645740747451783115) ) ) {
                result[0] += 0.02847939089424923;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                  result[0] += 0.023304426307299706;
                } else {
                  result[0] += -0.018185904229529076;
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.021042120549595617;
              } else {
                result[0] += 0.02713487535601332;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.022161789095258105;
                      } else {
                        result[0] += -0.04722908428264511;
                      }
                    } else {
                      result[0] += -0.03983414714749929;
                    }
                  } else {
                    result[0] += 0.058672222685769965;
                  }
                } else {
                  result[0] += -0.03546429982426292;
                }
              } else {
                result[0] += -0.050139513335272806;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.03713270596072596;
              } else {
                result[0] += 0.034935502997613326;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.0030120082565942524;
              } else {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.07237093988382585;
                } else {
                  result[0] += 0.012809387080915006;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                    result[0] += -0.04564720864030052;
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.026188787827089412;
                    } else {
                      result[0] += 0.017684579104598634;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.11119195842444017;
                  } else {
                    result[0] += -0.047836646274500905;
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    result[0] += -0.025156747224067424;
                  } else {
                    result[0] += 0.037638305162020436;
                  }
                } else {
                  result[0] += -0.04247062769464547;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.062178432391025384;
                } else {
                  result[0] += 0.005255197417747087;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.0270595049397093;
                } else {
                  result[0] += -0.07298793944606329;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.016154449601638598;
                } else {
                  result[0] += -0.08897593192188905;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.0025199317507130295;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.138696432113648349) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += 0.0362874909160722;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.044813199879419854;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                          result[0] += 0.022960109438269264;
                        } else {
                          result[0] += -0.06187479759432657;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                        result[0] += 0.06984109086899641;
                      } else {
                        result[0] += 0.010694530208668632;
                      }
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.012819601282177826;
                      } else {
                        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.05731371796063731;
                          } else {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.03673622484694384;
                              } else {
                                result[0] += -0.14667736035643844;
                              }
                            } else {
                              result[0] += 0.05627487320706234;
                            }
                          }
                        } else {
                          result[0] += 0.0802921209824145;
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
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.075335502624512607) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.02771647937178902;
        } else {
          result[0] += -0.07244407159209952;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
            result[0] += -0.041507821188129956;
          } else {
            result[0] += 0.03608635375220588;
          }
        } else {
          result[0] += -0.03137943453138832;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.021599526210849877;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.006713707744927644;
            } else {
              result[0] += -0.042986811313370295;
            }
          }
        } else {
          result[0] += -0.06235522730355786;
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.004279299692546857;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.038294641602036855;
            } else {
              result[0] += -0.08637979063350805;
            }
          } else {
            result[0] += -0.08371349146789549;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
        result[0] += 0.02241307360182533;
      } else {
        result[0] += -0.00747343961142301;
      }
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.176905632019043857) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.024302803842558965;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03804707647446432;
              } else {
                result[0] += 0.020704315007856036;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.008309992604736936;
              } else {
                result[0] += -0.027739148764160226;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
            result[0] += 0.0024277452979661815;
          } else {
            result[0] += -0.044681815981318276;
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.616744756698609287) ) ) {
                  result[0] += 0.027373956794017246;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += 0.021556488072012353;
                  } else {
                    result[0] += -0.043536039039313766;
                  }
                }
              } else {
                result[0] += 0.0461663126559477;
              }
            } else {
              result[0] += -0.0011708975529057164;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0032992334084314748;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += 0.042763796447685104;
                  } else {
                    result[0] += -0.06334728735138122;
                  }
                } else {
                  result[0] += -0.0164208984703381;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.031054403932216097;
              } else {
                result[0] += 0.030347912255203082;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.0031093967129004647;
              } else {
                result[0] += -0.040817438717928214;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.018874810664494562;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.09563976441005702;
                  } else {
                    result[0] += -0.04815611005767592;
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                    result[0] += -0.03142961001090236;
                  } else {
                    result[0] += 0.03149850530404987;
                  }
                } else {
                  result[0] += -0.05049186230917673;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.009722650229083274;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.03134464902149583;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.07897547693422796;
                      } else {
                        result[0] += -0.03791842852073747;
                      }
                    } else {
                      result[0] += 0.013310170170493435;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.021731381451159132;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.024214080885649497;
                  } else {
                    result[0] += -0.06745214105020725;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03599121729955092;
                } else {
                  result[0] += -0.009409074882595639;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06499592916229498;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.05749429303351941;
                      } else {
                        result[0] += 0.02828790365782626;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.012499983411349435;
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.005035681525803404;
                        } else {
                          result[0] += 0.05930002002852999;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += -0.1064939581505985;
                        } else {
                          result[0] += -0.009968843677233727;
                        }
                      } else {
                        result[0] += 0.039664851735759746;
                      }
                    }
                  }
                } else {
                  result[0] += 0.007873319245629616;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.615975379943848544) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
            result[0] += -0.06321389348543048;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.0318875699044979;
            } else {
              result[0] += 0.008822214231597283;
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.007068566738009402;
          } else {
            result[0] += 0.0599727445158382;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.0431429974874037;
        } else {
          result[0] += -0.10621967186658321;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.05026910279224374;
          } else {
            result[0] += -0.055483464970829316;
          }
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.08241022228920461;
            } else {
              result[0] += -0.008106331157637968;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.05846844546926036;
                } else {
                  result[0] += -0.02023879233105841;
                }
              } else {
                result[0] += -0.04173805240677904;
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.03319040420221111;
              } else {
                result[0] += -0.041493906180321116;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
            result[0] += 0.023754635273272984;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
              result[0] += -0.00744018311482971;
            } else {
              result[0] += -0.07056160667886939;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0020188429346347475;
            } else {
              result[0] += -0.06129846569501779;
            }
          } else {
            result[0] += 0.019492694417991386;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.028787555246677116;
          } else {
            result[0] += -0.07280897115901934;
          }
        } else {
          result[0] += -0.024428361849054194;
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += -0.045618054710655116;
        } else {
          result[0] += -0.07734388200038722;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.05282646754580933;
                } else {
                  result[0] += -0.00289210751383722;
                }
              } else {
                result[0] += -0.04104244160565246;
              }
            } else {
              result[0] += -0.023805222825691454;
            }
          } else {
            result[0] += -0.062790068120863;
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.018862467685151988;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                  result[0] += -0.0028669857903390198;
                } else {
                  result[0] += -0.03678949629890838;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.02370917400008414;
                } else {
                  result[0] += 0.021795388889672418;
                }
              }
            }
          } else {
            result[0] += -0.023261959905390886;
          }
        }
      } else {
        result[0] += 0.0025041992672134197;
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.0029353321056321408;
              } else {
                result[0] += -0.04150163823652847;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.020800652211056446;
              } else {
                result[0] += 0.0033718179486281713;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.02263884410784618;
                  } else {
                    result[0] += 0.0379055518491994;
                  }
                } else {
                  result[0] += -0.07140297236849368;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.04467582195526906;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.03662225314578535;
                        } else {
                          result[0] += 0.11523879196718936;
                        }
                      } else {
                        result[0] += -0.026179193673344883;
                      }
                    } else {
                      result[0] += 0.0366891129387836;
                    }
                  } else {
                    result[0] += 0.0360885127986393;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
                    result[0] += 0.027418055984500974;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.07809149056487598;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.01618314804974757;
                      } else {
                        result[0] += 0.045098603894211386;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0710943902342451;
                  } else {
                    result[0] += 0.06441596253625174;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.06573013861444416;
                  } else {
                    result[0] += 0.015103975260980996;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.07750303246130262;
                  } else {
                    result[0] += -0.01451931772075042;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                result[0] += -0.06473231162697761;
              } else {
                result[0] += 0.008056266483750775;
              }
            } else {
              result[0] += -0.08701674167278003;
            }
          } else {
            result[0] += 0.024924862224254402;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.010776564115250228;
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += 0.007145479071771846;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
              result[0] += 0.013811343605954922;
            } else {
              result[0] += -0.06084496432545107;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
      result[0] += 0.0214754886735911;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
        result[0] += -0.020208948136809585;
      } else {
        result[0] += -0.052450614099315424;
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += 0.023761784285298827;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                      result[0] += 0.028139198492375807;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.00119127628064239;
                      } else {
                        result[0] += -0.05485545073709647;
                      }
                    }
                  }
                } else {
                  result[0] += 0.049204115242565544;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.049568218761049936;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                    result[0] += -0.06652905885546237;
                  } else {
                    result[0] += 0.012928340453509582;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                result[0] += 0.014186045183825817;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.0521007109095745;
                  } else {
                    result[0] += 0.011445099692272552;
                  }
                } else {
                  result[0] += 0.03510032678648111;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.005650958010150057;
              } else {
                result[0] += -0.07033261115847005;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  result[0] += 0.017138587641174697;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.07787298507441211;
                  } else {
                    result[0] += -0.02050443661359864;
                  }
                }
              } else {
                result[0] += 0.02811068229591749;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0007563519960707656;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.20949268341064631) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += 0.04774010361243716;
                  } else {
                    result[0] += -0.027932892292852337;
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.018187894296770726;
                    } else {
                      result[0] += -0.06698784212017007;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.08263221445316753;
                      } else {
                        result[0] += 0.01795963190476857;
                      }
                    } else {
                      result[0] += -0.03407511914966418;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.03804359937871333;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += 0.025081265327929106;
                  } else {
                    result[0] += -0.036113368101187135;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                result[0] += -0.005877282428666496;
              } else {
                result[0] += -0.04721768055850301;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.014274088257096472;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.004736339131713471;
            } else {
              result[0] += -0.06534050137771266;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.044031295550342464;
              } else {
                result[0] += -0.00709334310645504;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.006322301768772654;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
                  result[0] += -0.042769118901785225;
                } else {
                  result[0] += 0.006860812135123061;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0051672112116459504;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.02663621591475429;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.05022166573107004;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                      result[0] += -0.02535071973359236;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += -0.006818347340015011;
                      } else {
                        result[0] += 0.051347130950423196;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                  result[0] += -0.03192752880845284;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
                        result[0] += 0.04404365544276205;
                      } else {
                        result[0] += -0.00563941842119665;
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.524927973747253862) ) ) {
                        result[0] += 0.048579357898858186;
                      } else {
                        result[0] += -0.04846498903406937;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                      result[0] += -0.02586022259528824;
                    } else {
                      result[0] += 0.02476590700558909;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.463808774948121005) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
          result[0] += -0.0590835236020567;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81804704666137873) ) ) {
            result[0] += -0.04363170225236169;
          } else {
            result[0] += 0.014864365896141058;
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.1094091748894694;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
            result[0] += -0.047310131340357706;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.038992607797559004;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += -0.08217724949896711;
              } else {
                result[0] += 0.04403356854194922;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.03566360632755067;
          } else {
            result[0] += -0.04431892063741537;
          }
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.07622046314169731;
            } else {
              result[0] += 0.006560917209545299;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.03704037964589787;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.027406786780307692;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.02420620384307744;
                  } else {
                    result[0] += -0.048190084839556196;
                  }
                }
              } else {
                result[0] += -0.03536114747029441;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                result[0] += 0.0420405338556545;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.00126463161800071;
                } else {
                  result[0] += -0.07380070703445331;
                }
              }
            } else {
              result[0] += 0.030432573652052194;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += -0.0604664781439285;
            } else {
              result[0] += 0.004277408278562267;
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
              result[0] += 0.01932551855952755;
            } else {
              result[0] += -0.09130468346633754;
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.024472048788067424;
              } else {
                result[0] += -0.034705540269348925;
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.036492575545074006;
              } else {
                result[0] += 0.01885197168631904;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.043464688261231114;
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.022494747297933427;
                } else {
                  result[0] += 0.0754272953500511;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.722943305969239169) ) ) {
                  result[0] += 0.1075202246536125;
                } else {
                  result[0] += -0.10915078525973088;
                }
              }
            } else {
              result[0] += -0.0712685621058146;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.037115087925917216;
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.01086139402355862;
              } else {
                result[0] += -0.1284517127671081;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.08442463803153673;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.021125542358467933;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.024233319217846012;
                } else {
                  result[0] += -0.1073575640825627;
                }
              }
            }
          } else {
            result[0] += -0.05805417398787506;
          }
        } else {
          result[0] += -0.0722633140510449;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.02539986632944921;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.802100181579590732) ) ) {
                result[0] += 0.002020331364894162;
              } else {
                result[0] += -0.0382036290152927;
              }
            } else {
              result[0] += 0.0036999642036944797;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
            result[0] += -0.013235249477918135;
          } else {
            result[0] += -0.046867192864033486;
          }
        }
      } else {
        result[0] += 0.001776700491678462;
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.469231128692627841) ) ) {
          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.0033559552821205763;
                } else {
                  result[0] += -0.0456353861086955;
                }
              } else {
                result[0] += 0.013964353521651535;
              }
            } else {
              result[0] += 0.026593118400140406;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.777633190155030185) ) ) {
                result[0] += -0.0005283960420033389;
              } else {
                result[0] += 0.011113305766784615;
              }
            } else {
              result[0] += -0.034134358044351144;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.020806065466492683;
                } else {
                  result[0] += -0.03297984459363334;
                }
              } else {
                result[0] += -0.07337221149402222;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.003838300704956943) ) ) {
                  result[0] += 0.006224016699478679;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.044970084436515886;
                  } else {
                    result[0] += -0.048122440468112655;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                  result[0] += 0.007537381777930872;
                } else {
                  result[0] += 0.05043953590080013;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.0207071299964147;
              } else {
                result[0] += -0.02487199994314901;
              }
            } else {
              result[0] += 0.03657806887381797;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.01021164883674849;
        } else {
          result[0] += 0.005612288684127831;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.04552340544900471;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.768316030502320224) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.008561488431852348;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
              result[0] += 0.0011204963197239662;
            } else {
              result[0] += -0.04064613036890458;
            }
          }
        } else {
          result[0] += -0.03566346272648429;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.045986565537051466;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.04070013605954325;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
              result[0] += -0.05604520638893758;
            } else {
              result[0] += 0.03258447105357316;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += -0.04012259123231608;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
            result[0] += -0.05687503027122995;
          } else {
            result[0] += -0.09893116208241842;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
              result[0] += 0.010394006576903082;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.01165466901869986;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                  result[0] += -0.05421451516212622;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.03160454804685053;
                  } else {
                    result[0] += 0.004941529171144279;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.011428596133137172;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.033759082738684026;
              } else {
                result[0] += -0.10033428542809993;
              }
            }
          }
        } else {
          result[0] += -0.04946094462110271;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.07281943657914239;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.847873449325562412) ) ) {
                result[0] += -0.0005296843162629643;
              } else {
                result[0] += -0.036752077442599694;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                    result[0] += 0.03143589624899281;
                  } else {
                    result[0] += -0.02923153248351075;
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.0004632689682512299;
                        } else {
                          result[0] += -0.07224323421963563;
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.06320216743155459;
                          } else {
                            result[0] += 0.00021225180849989138;
                          }
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.00024065373460959957;
                          } else {
                            result[0] += -0.06562806124432932;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.07815584414952957;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.008338999790864573;
                    } else {
                      result[0] += 0.03158023957508119;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.04426677235575989;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                    result[0] += -0.05067255587761981;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                        result[0] += -0.03300059242160856;
                      } else {
                        result[0] += 0.05832072333675379;
                      }
                    } else {
                      result[0] += -0.011320213371200503;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.057013021732530235;
              } else {
                result[0] += 0.010674885800109788;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.038782485314526716;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += 0.0630354116298319;
                  } else {
                    result[0] += -0.04109603816377105;
                  }
                } else {
                  result[0] += 0.018876102056178883;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.09784601076048106;
              } else {
                result[0] += -0.03284369945092479;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.7619357109069842) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.046186947069671434;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.02945468056954352;
                  } else {
                    result[0] += -0.02512551153241744;
                  }
                }
              } else {
                result[0] += 0.1104822247827863;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.0117690222472935;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += 0.010031908570480948;
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.09744179852674172;
                } else {
                  result[0] += -0.021579442422914624;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                  result[0] += -0.004404369302371586;
                } else {
                  result[0] += 0.05417959011009024;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.00793088505620205;
          } else {
            result[0] += 0.005049080494464582;
          }
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
            result[0] += -0.09313955086959556;
          } else {
            result[0] += 0.054248245542157264;
          }
        } else {
          result[0] += -0.10424081814377739;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.03392104576820002;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.04199138195212022;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += -0.0022763416907901303;
            } else {
              result[0] += -0.08795073504437509;
            }
          } else {
            result[0] += -0.0014221451988585382;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.00039345254547150344;
                } else {
                  result[0] += -0.035931714123375695;
                }
              } else {
                result[0] += -0.06020379259439095;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.04021659117776792;
              } else {
                result[0] += -0.006621481120591296;
              }
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.241249561309815341) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.019972441184575866;
                } else {
                  result[0] += -0.09139412232222015;
                }
              } else {
                result[0] += 0.05961259440730076;
              }
            } else {
              result[0] += -0.06630393098578134;
            }
          }
        } else {
          result[0] += -0.07502202515382855;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.07172877184199826;
        } else {
          result[0] += -0.03744996334491164;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.017363068390275495;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.036670446395874912) ) ) {
                result[0] += 0.0049508983152721;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  result[0] += -0.04723864882317811;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.03155018054507368;
                      } else {
                        result[0] += 0.047621860383741076;
                      }
                    } else {
                      result[0] += -0.04474801344436635;
                    }
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.027590695667503315;
                    } else {
                      result[0] += 0.0334814253688431;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.009704251537829618;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.031592420758795904;
              } else {
                result[0] += -0.09748076915914639;
              }
            }
          }
        } else {
          result[0] += -0.044983448767711064;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.07015558656836433;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                  result[0] += 0.017516362986060855;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.041251209951377224;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.005205376794968513;
                      } else {
                        result[0] += 0.03431980242830187;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.04213198331335658;
                    } else {
                      result[0] += -0.0020995188171619755;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.0872417144180826;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.07175706558069073;
                    } else {
                      result[0] += -0.02086775453431551;
                    }
                  } else {
                    result[0] += 0.05613291992584599;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                result[0] += -0.039452424120688674;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.026142683635578867;
                    } else {
                      result[0] += -0.06564804893798726;
                    }
                  } else {
                    result[0] += 0.009396253537810354;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                    result[0] += 0.006082842714130236;
                  } else {
                    result[0] += 0.04343180123655243;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.0062521174239620215;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.09309160810436595;
              } else {
                result[0] += -0.02990865940218301;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.448888063430787021) ) ) {
                result[0] += -0.011181076063870843;
              } else {
                result[0] += 0.08653755668052417;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.07801548816201959;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                result[0] += 0.013549144603783146;
              } else {
                result[0] += -0.052222830166218426;
              }
            }
          } else {
            result[0] += -0.0019508245633270286;
          }
        } else {
          result[0] += 0.008326563298353297;
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.0028617955497026774;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0911903381347674) ) ) {
                  result[0] += -0.0066634368328900885;
                } else {
                  result[0] += -0.04137476942788379;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.04509655301833094;
                } else {
                  result[0] += -0.008824695868934054;
                }
              }
            } else {
              result[0] += -0.05459819459631972;
            }
          }
        } else {
          result[0] += 0.003916754315535629;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      result[0] += -0.011106668711363699;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
        result[0] += -0.004017288571368137;
      } else {
        result[0] += -0.04962233871978757;
      }
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += 0.020055274297180563;
          } else {
            result[0] += -2.7552639024025484e-05;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.03742229349754172;
            } else {
              result[0] += 0.017799037809887296;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.05611697629339901;
              } else {
                result[0] += -0.011834460156843432;
              }
            } else {
              result[0] += -0.008701882455685747;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.030047392298679323;
              } else {
                result[0] += -0.013969607645161732;
              }
            } else {
              if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.36324071884155451) ) ) {
                      result[0] += -0.03468872104752553;
                    } else {
                      result[0] += 0.0452064366885854;
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.08328391890739219;
                    } else {
                      result[0] += 0.027719876360218776;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                      result[0] += 0.00782068219958054;
                    } else {
                      result[0] += -0.04500424209793394;
                    }
                  } else {
                    result[0] += 0.022619211745302292;
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                    result[0] += -0.062270672610294044;
                  } else {
                    result[0] += -0.0007821733085137527;
                  }
                } else {
                  result[0] += -0.07041343374561432;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.040157539481145615;
              } else {
                result[0] += -0.09470423393593613;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.584782838821412021) ) ) {
                  result[0] += -0.07351588240392477;
                } else {
                  result[0] += -0.005355936878039608;
                }
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.04652295403272594;
                    } else {
                      result[0] += -0.10073354995127051;
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.11337632855080466;
                    } else {
                      result[0] += 0.02662489531046372;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.015199745981240496;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.28202676773071467) ) ) {
                      result[0] += -0.05562701087384651;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.027182368907131146;
                      } else {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.1359241730718277;
                        } else {
                          result[0] += 0.00446296675264994;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                result[0] += 0.004932101871357013;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.001236519517144633;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.0025928916769596263;
                  } else {
                    result[0] += -0.052272234407294095;
                  }
                }
              }
            } else {
              result[0] += -0.06250366142790088;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.03634358944338504;
                } else {
                  result[0] += 0.00557158614881232;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
                    result[0] += -0.045160436238024126;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                      result[0] += -0.03154143637674069;
                    } else {
                      result[0] += 0.022282808345209335;
                    }
                  }
                } else {
                  result[0] += 0.023420361592614684;
                }
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.009435757223248737;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += -0.028947475421536924;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.010047117729143113;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += 0.037882608925685554;
                      } else {
                        result[0] += 0.09765685416436803;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.07449093556555142;
                  } else {
                    result[0] += -0.017452618737034357;
                  }
                } else {
                  result[0] += 0.017815374275311886;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
          result[0] += -0.06488891318254672;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.07248727439816403;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.03248290963989047;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.138696432113648349) ) ) {
                result[0] += -0.023703205417760765;
              } else {
                result[0] += 0.047876825861509355;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.01922335644149732;
          } else {
            result[0] += -0.05863859069745606;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.349460363388062412) ) ) {
            result[0] += 0.014800331166791837;
          } else {
            result[0] += 0.0682604016017986;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.005489438987478929;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          result[0] += -0.01763033990429294;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += -0.022367963071343092;
          } else {
            result[0] += -0.0558193229973324;
          }
        }
      }
    } else {
      result[0] += 0.004133960369547459;
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.615975379943848544) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
              result[0] += 0.00035704494101013846;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                  result[0] += 0.014294485565342374;
                } else {
                  result[0] += -0.05328068237850223;
                }
              } else {
                result[0] += 0.008145709653228235;
              }
            }
          } else {
            result[0] += 0.006193955115421642;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            result[0] += 0.009724672157268861;
          } else {
            result[0] += -0.045494190940464696;
          }
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.512487888336182529) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.015044851078093766;
                      } else {
                        result[0] += 0.0375289889991936;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                        result[0] += -0.03716010945052947;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += -0.020812291880732622;
                        } else {
                          result[0] += 0.02662374149693399;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.013941329124206193;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    result[0] += 0.018615936470092447;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                        result[0] += -0.013794040744466466;
                      } else {
                        result[0] += -0.05500213776671275;
                      }
                    } else {
                      result[0] += 0.04305448338120792;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                    result[0] += -0.0703075352168535;
                  } else {
                    result[0] += -0.024854805587815568;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += -0.030512175279913834;
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.020921280676600075;
                    } else {
                      result[0] += -0.00726230061021706;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.161602735519410068) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.0012338903225064906;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0892543998546481;
                    } else {
                      result[0] += -0.02940328648149758;
                    }
                  }
                } else {
                  result[0] += 0.04756939582891803;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  result[0] += -0.034727301421696666;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.04157333400532709;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.052946563096716304;
                        } else {
                          result[0] += -0.021296389903502393;
                        }
                      } else {
                        result[0] += 0.044471204563653284;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.913499355316162998) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.018490667233596953;
                        } else {
                          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.015199343626823892;
                          } else {
                            result[0] += -0.13094652311095595;
                          }
                        }
                      } else {
                        result[0] += 0.036574006133516884;
                      }
                    } else {
                      result[0] += 0.04406399009715683;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                  result[0] += 0.05448788907216842;
                } else {
                  result[0] += 0.002539258908159625;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
                    result[0] += 0.005776081110026286;
                  } else {
                    result[0] += -0.0738371279095687;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.06820132423381381;
                      } else {
                        result[0] += 0.008674732125488534;
                      }
                    } else {
                      result[0] += 0.026630964650999;
                    }
                  } else {
                    result[0] += 0.026281621827946074;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0405922492628579;
                } else {
                  result[0] += 0.0012974359951635785;
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.08500747209645437;
                } else {
                  result[0] += -0.009242163401042077;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.02766371934495368;
            } else {
              result[0] += -0.10112827991725243;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
              result[0] += -0.04633567361560239;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
                result[0] += -0.008114527848887117;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06946712116809517;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                    result[0] += -0.02180481010027813;
                  } else {
                    result[0] += 0.04252376540978878;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          result[0] += -0.08905248876211985;
        } else {
          result[0] += 0.049515741583481744;
        }
      } else {
        result[0] += -0.10345221786121615;
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += 0.004606382380697746;
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.002393994487181357;
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += 0.08233635976509569;
              } else {
                result[0] += -0.04748988509163853;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.701612949371339667) ) ) {
                result[0] += -0.0009241725012715026;
              } else {
                result[0] += 0.06000290383118978;
              }
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.07081807779706045;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.022268755727940525;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.12236662745203089;
                  } else {
                    result[0] += -0.091422147854734;
                  }
                }
              }
            } else {
              result[0] += -0.08817242874271033;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.03376302769121764;
        } else {
          result[0] += -0.06944004175318429;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += -0.0024848415855046753;
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.009588474279592217;
          } else {
            result[0] += -0.03580202420178546;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0011424404474171886;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += -0.05306207290939337;
            } else {
              result[0] += 0.04613121559811069;
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.025752622811525783;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.13715192495763667;
                } else {
                  result[0] += 0.007157346651596855;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.04111482549457283;
              } else {
                result[0] += 0.03298492993032623;
              }
            }
          } else {
            result[0] += -0.07727347020163476;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.07457487319527038;
          } else {
            result[0] += -0.004260999103793953;
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.019902399416417405;
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                          result[0] += -0.05479738345677078;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.587220668792725498) ) ) {
                            result[0] += -0.002984343936822613;
                          } else {
                            result[0] += 0.035975552475611496;
                          }
                        }
                      } else {
                        result[0] += -0.07887938405405859;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                            result[0] += 0.016390519915925182;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.725620865821838823) ) ) {
                              result[0] += 0.03259690473442661;
                            } else {
                              result[0] += -0.020860856551771292;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                            result[0] += -0.004414227868160599;
                          } else {
                            result[0] += 0.04216604744231608;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.03826488269402313;
                        } else {
                          result[0] += -0.12061371499556717;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.333273410797120029) ) ) {
                        result[0] += 0.0004441843067568758;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.018829370014219634;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                            result[0] += 0.04811246510351391;
                          } else {
                            result[0] += -0.08469057037793959;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                    result[0] += 0.0017281369484959703;
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                      result[0] += 0.02110579721770703;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                        result[0] += 0.02759790730717819;
                      } else {
                        result[0] += -0.04608631390870095;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                  result[0] += -0.026856435097422027;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += -0.03644354222292381;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.049958241251547936;
                    } else {
                      result[0] += -0.0008959043601901947;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.015617270588684671;
            }
          } else {
            if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.06926930048014808;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
                  result[0] += -0.04757705057365172;
                } else {
                  result[0] += 0.010042967814117018;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                result[0] += -0.009250272668704065;
              } else {
                result[0] += 0.04241462706602908;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80124759674072443) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.689592361450196201) ) ) {
                result[0] += 0.0010503521895667443;
              } else {
                result[0] += -0.030815184149461113;
              }
            } else {
              result[0] += 0.02667281051736339;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              result[0] += -0.009697554030496242;
            } else {
              result[0] += -0.04093193559517052;
            }
          }
        } else {
          result[0] += 0.003107858211233576;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY(  (data[48].missing != -1) && (data[48].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.018270240829534386;
            } else {
              result[0] += 0.09983715719257205;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.023566054166954196;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.02659204931824844;
              } else {
                result[0] += -0.036100071044985714;
              }
            }
          }
        } else {
          result[0] += -0.06037164384571933;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.017626814756202213;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.349460363388062412) ) ) {
              result[0] += -0.024095791744639522;
            } else {
              result[0] += 0.020198031957314386;
            }
          }
        } else {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += 0.004772786328565565;
          } else {
            result[0] += -0.02947322438517304;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.009319243983161944;
        } else {
          result[0] += -0.02340126441789516;
        }
      } else {
        result[0] += -0.05294357463999344;
      }
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.007126635473945356;
            } else {
              result[0] += 0.002028890761753439;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.010986246401130106;
                    } else {
                      result[0] += -0.055380789281532505;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0241548436554831;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                        result[0] += -0.029539455202030697;
                      } else {
                        result[0] += 0.04313945581845027;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
                          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                              result[0] += 0.008733240103648185;
                            } else {
                              result[0] += -0.015303608497053693;
                            }
                          } else {
                            result[0] += 0.02698127697270756;
                          }
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.53498554229736506) ) ) {
                              result[0] += 0.02826482180714998;
                            } else {
                              result[0] += -0.036946974655220566;
                            }
                          } else {
                            result[0] += 0.0028727961567708623;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                          result[0] += 0.011121174041549164;
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.00269421997484285;
                            } else {
                              result[0] += -0.07992940566412693;
                            }
                          } else {
                            result[0] += 0.009799757050787894;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.689592361450196201) ) ) {
                          result[0] += 0.01189700537826802;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += 0.015626435948289396;
                            } else {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.308072090148926669) ) ) {
                                result[0] += -0.07158576874702642;
                              } else {
                                result[0] += -0.0013326396940824344;
                              }
                            }
                          } else {
                            result[0] += 0.015538253980769865;
                          }
                        }
                      } else {
                        result[0] += -0.017511467562336094;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.04883574981845088;
                    } else {
                      result[0] += 0.0009678622492117439;
                    }
                  }
                }
              } else {
                result[0] += 0.014752512264137744;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.008796313686901595;
              } else {
                result[0] += 0.002293722502376723;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.01708788621819765;
            } else {
              result[0] += -0.09996943513154649;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.018179248009086057;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                result[0] += 0.034903459855670387;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.041860702701099796;
                } else {
                  result[0] += 0.08815812180206184;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0195939089474101;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.12872368355805294;
              } else {
                result[0] += 0.021622005876295567;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.03668389641148118;
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.020412904962813037;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
                  result[0] += 0.02265332013248963;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.01138236820710867;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.0210540827734276;
                    } else {
                      result[0] += 0.11292621505596163;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
              result[0] += -0.090440676855121;
            } else {
              result[0] += 0.034390398072417463;
            }
          } else {
            result[0] += -0.10739783411861037;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        result[0] += -0.10163735081084846;
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
          result[0] += -0.08812740920501853;
        } else {
          result[0] += 0.04042391905856588;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      result[0] += -0.004187497658555967;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
        result[0] += -0.01318896615803955;
      } else {
        result[0] += -0.05047939402261939;
      }
    }
  } else {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
          result[0] += 0.0003804618171259926;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.006259860963153047;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.03038145052589914;
              } else {
                result[0] += -0.05676142625606337;
              }
            } else {
              result[0] += -0.0013974640533350555;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.512487888336182529) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.014775645693693232;
                    } else {
                      result[0] += 0.039546828460328795;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                      result[0] += -0.0357881971502547;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.016132460057123394;
                      } else {
                        result[0] += -0.0345550647513873;
                      }
                    }
                  }
                } else {
                  result[0] += -0.00964445623868843;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                  result[0] += 0.015066261169997073;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.04779248650439714;
                    } else {
                      result[0] += 0.0028069687005147113;
                    }
                  } else {
                    result[0] += 0.037504297683352235;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.20949268341064631) ) ) {
                  result[0] += -0.042488864612356264;
                } else {
                  result[0] += -0.004711098250302576;
                }
              } else {
                result[0] += 0.003535387044973912;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.019229173660279208) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.001683962750675775;
                } else {
                  result[0] += -0.040218754567085235;
                }
              } else {
                result[0] += 0.03698223554298991;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.020764957391414453;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.049987526397505366;
                  } else {
                    result[0] += 0.010499556282910542;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.061580230423969644;
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.047694185099948466;
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.04660917826674896;
                        } else {
                          result[0] += -0.013227772189516497;
                        }
                      } else {
                        result[0] += 0.04615657470956561;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.020127415657043901) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.561026811599732333) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11192369461059748) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += 0.03188969300266372;
                          } else {
                            result[0] += -0.03871733058374577;
                          }
                        } else {
                          result[0] += -0.05983569259441651;
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.0484536861889413;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.04147383629927523;
                          } else {
                            result[0] += 0.028106577508072734;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.00389740183086224;
                      } else {
                        result[0] += 0.05700272363984124;
                      }
                    }
                  } else {
                    result[0] += -0.037293578059994524;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              result[0] += 0.03730700178978432;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += 0.06896165506308102;
                  } else {
                    result[0] += -0.02497736622184473;
                  }
                } else {
                  result[0] += -0.08508475031947138;
                }
              } else {
                result[0] += 0.01130344737421616;
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.051201192926253705;
            } else {
              if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.06845583425131967;
                } else {
                  result[0] += -0.015558203226720755;
                }
              } else {
                result[0] += 0.007188133409866651;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.007551316843422722;
          } else {
            result[0] += -0.04098365385048509;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
            result[0] += -0.03833006948201666;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
              result[0] += -0.0008896685780076243;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.05765073958829732;
              } else {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.09032605253078152;
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += -0.13667432407259758;
                    } else {
                      result[0] += -0.006331732698651194;
                    }
                  } else {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.060995467470148085;
                      } else {
                        result[0] += -0.011030117378096395;
                      }
                    } else {
                      result[0] += 0.05582912960484823;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.033222635395687136;
        } else {
          result[0] += -0.09937350920468996;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.04105321745215207;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
        result[0] += -0.00023304065654494914;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.768316030502320224) ) ) {
            result[0] += -0.00254487489615168;
          } else {
            result[0] += -0.037382471074554176;
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += -0.05265074407451468;
          } else {
            result[0] += 0.05521618622747884;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.616744756698609287) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.017336090913789588;
              } else {
                result[0] += 0.04347979574667236;
              }
            } else {
              result[0] += -0.009280210714004382;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
              result[0] += 0.009619376346354305;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.05424205129486429;
                } else {
                  result[0] += 0.02534136422396714;
                }
              } else {
                result[0] += 0.020776482875151325;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0037784369211361878;
            } else {
              result[0] += -0.06410607715717334;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += 0.01006049718342434;
              } else {
                result[0] += -0.06386457971263357;
              }
            } else {
              result[0] += 0.021252302015944382;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.03466883544422316;
                } else {
                  result[0] += -0.008155450144381153;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.04283126586300626;
                } else {
                  result[0] += -0.025149885144731616;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                  result[0] += -0.0848933914978542;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
                    result[0] += -0.07104508449843668;
                  } else {
                    result[0] += 0.022262174812084368;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.343781709671021396) ) ) {
                  result[0] += -0.12439785796481524;
                } else {
                  result[0] += -0.038610709522153605;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.85305833816528498) ) ) {
                result[0] += 0.004727927749096056;
              } else {
                result[0] += 0.047191258736110936;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                result[0] += 0.007063860814303664;
              } else {
                result[0] += -0.03541470384332953;
              }
            }
          }
        } else {
          result[0] += -0.0011178541250031423;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.616744756698609287) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                result[0] += -0.016533673211288843;
              } else {
                result[0] += -0.04776558496018245;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                result[0] += -0.016852571319944137;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.028423779482725745;
                } else {
                  result[0] += -0.024208667876835947;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.02753950063666087;
            } else {
              result[0] += -0.017983395358691645;
            }
          }
        } else {
          result[0] += -0.07904931104529578;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.012681621646863761;
          } else {
            result[0] += -0.026445883482627483;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.017862282121190096;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.07720435618306672;
                } else {
                  result[0] += -0.03230488921631989;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
                  result[0] += -0.02136480956477449;
                } else {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.005557144198828102;
                  } else {
                    result[0] += 0.032994935142861975;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += 0.020082741756109367;
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.04991380717144574;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.019952447050697302;
                  } else {
                    result[0] += 0.03293196380827584;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.524927973747253862) ) ) {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.017444644194141833;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.060618005133503954;
                      } else {
                        result[0] += -0.004505000300471373;
                      }
                    } else {
                      result[0] += 0.10273444913913748;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.16267596889149621;
                        } else {
                          result[0] += -0.009462835944206336;
                        }
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.07273294028336239;
                        } else {
                          result[0] += 0.0422619021285343;
                        }
                      }
                    } else {
                      result[0] += 0.032900072340790086;
                    }
                  } else {
                    result[0] += -0.05476275541199293;
                  }
                }
              } else {
                result[0] += -0.048110844952463976;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.860674262046814409) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += 0.03508594898965913;
          } else {
            result[0] += -0.03814531497691961;
          }
        } else {
          result[0] += 0.029825903164100172;
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += -0.00803007385874571;
          } else {
            result[0] += -0.07846467021676196;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.737386107444763628) ) ) {
            result[0] += -0.018495474302624153;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.06939468722674336;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.08736198458967959;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
                    result[0] += 0.018341228748901765;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.0773472404296652;
                    } else {
                      result[0] += 0.07309249702386296;
                    }
                  }
                } else {
                  result[0] += -0.06723234908128294;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.07107953904810832;
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
              result[0] += 0.012924021858668103;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.010156153953049554;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.02358743000357833;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.08361385160070439;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                        result[0] += -0.04738348366787237;
                      } else {
                        result[0] += 0.04621499236166769;
                      }
                    }
                  } else {
                    result[0] += -0.012501880585304088;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.007529791712689925;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.025263510337548596;
              } else {
                result[0] += -0.09052446032096773;
              }
            }
          }
        } else {
          result[0] += -0.037568052118822255;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
          result[0] += 0.0022073706821175054;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
            result[0] += -0.05083571212358845;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.0412731696215835;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.10610209426642457;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.08983198669740342;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                      result[0] += -0.04800883713719406;
                    } else {
                      result[0] += 0.019232365890641114;
                    }
                  }
                } else {
                  result[0] += -0.058678818984293;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.767332553863526279) ) ) {
            result[0] += -0.008957700588089562;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.013040452078844753;
            } else {
              result[0] += 0.025637993706919396;
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0002686558786296518;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.014788627624512607) ) ) {
                        result[0] += 0.008292791212824007;
                      } else {
                        result[0] += 0.02684168289933735;
                      }
                    }
                  } else {
                    result[0] += 0.031220432621167707;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
                      result[0] += 0.020026685486341806;
                    } else {
                      result[0] += -0.04005329493474868;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                      result[0] += -0.005707717063298082;
                    } else {
                      result[0] += 0.013608873632202671;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                      result[0] += 0.023293184820159088;
                    } else {
                      result[0] += -0.013676150814663699;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                      result[0] += -0.0007851598754413476;
                    } else {
                      result[0] += -0.06204923883368716;
                    }
                  }
                } else {
                  result[0] += 0.015453338863478142;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.04468947042764069;
                    } else {
                      result[0] += 0.04789783210500732;
                    }
                  } else {
                    result[0] += -0.037863642868458655;
                  }
                } else {
                  result[0] += -0.01712290387040764;
                }
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.011291419116140332;
                    } else {
                      result[0] += -0.05166523364808049;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.06003009638982683;
                    } else {
                      result[0] += 0.10339750999908387;
                    }
                  }
                } else {
                  result[0] += 0.009882006643858363;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                result[0] += 0.02596973024503989;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 0.010026779680243935;
                } else {
                  result[0] += -0.05082136485556307;
                }
              }
            } else {
              result[0] += 0.026346362318896244;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.005917910139663251;
        } else {
          result[0] += 0.00326055098278277;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0015920700527940817;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.02392353916955335;
              } else {
                result[0] += -0.021289374874626243;
              }
            } else {
              result[0] += -0.04561309007374477;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.04315291890833079;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.04051155981179428;
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.025469379264944143;
                } else {
                  result[0] += -0.04551671543019495;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.013906272788938072;
              } else {
                result[0] += -0.03969779419726929;
              }
            } else {
              result[0] += -0.06807131062269992;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)14.56755733489990412) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
            result[0] += -0.02969921731416362;
          } else {
            result[0] += -0.07116199663963169;
          }
        } else {
          result[0] += 0.07016665737067092;
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.942183732986451083) ) ) {
            result[0] += 0.013072419511150441;
          } else {
            result[0] += 0.03877414911430962;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
              result[0] += 0.046276230279531684;
            } else {
              result[0] += -0.054493461104788844;
            }
          } else {
            result[0] += 0.01014818598729591;
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
            result[0] += -0.059227019389123496;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.453179836273194248) ) ) {
              result[0] += -0.014776929569799133;
            } else {
              result[0] += 0.05236146493137647;
            }
          }
        } else {
          result[0] += -0.04817315997294023;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
              result[0] += 0.006993626828284661;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                result[0] += -0.0004196070984526447;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.04596398744650795;
                } else {
                  result[0] += -0.010073490997420251;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.005521873703568926;
              } else {
                result[0] += -0.022238267511765872;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.010808716376596974;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.0007419963509747431;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += -0.02663651045866601;
                    } else {
                      result[0] += 0.0357051017949277;
                    }
                  } else {
                    result[0] += -0.004489574739303784;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.01820248984146519;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
          result[0] += 0.0018756283681224187;
        } else {
          result[0] += -0.06084210778112627;
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += 0.0014769669229705935;
            } else {
              result[0] += 0.02156780916216474;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                result[0] += 0.0031325027909451563;
              } else {
                result[0] += -0.019663740784561318;
              }
            } else {
              result[0] += -0.01967520499081392;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.016934062950023383;
                } else {
                  result[0] += 0.020578983218289285;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.161602735519410068) ) ) {
                  result[0] += -0.06779890469799044;
                } else {
                  result[0] += 0.022339829370465927;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.037593843415955425;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.847873449325562412) ) ) {
                    result[0] += -0.0012588013855345111;
                  } else {
                    result[0] += -0.04029836212868859;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.048581593138797736;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                    result[0] += 0.00925670672173262;
                  } else {
                    result[0] += 0.042859605428083564;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.02968143522247277;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.06490100722644405;
                  } else {
                    result[0] += -0.006230662553563358;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.08337244129972521;
                  } else {
                    result[0] += -0.01410599554770335;
                  }
                } else {
                  result[0] += 0.0014776124189546072;
                }
              }
            } else {
              result[0] += 0.01827191913385915;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.006713570438997241;
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += 0.0038590807592931466;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += -0.009448988017589172;
            } else {
              result[0] += -0.08227356883214616;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += 0.004777400125286822;
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.003020914143809931;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
            result[0] += -0.020528076143013534;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.02243894283141227;
            } else {
              result[0] += -0.06177846547402837;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.008936741341832003;
        } else {
          result[0] += -0.028707209746143488;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.297262430191040927) ) ) {
                  result[0] += 0.013322313129737532;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.0006452187594989506;
                  } else {
                    result[0] += -0.05751020829199174;
                  }
                }
              } else {
                result[0] += 0.031075982082008846;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.314458370208742011) ) ) {
                result[0] += 0.0026379100847720867;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.05409851146571912;
                } else {
                  result[0] += 0.02192025144230929;
                }
              }
            }
          } else {
            result[0] += -0.09022405989634125;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.009724070397205797;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += 0.016875623775814474;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
                    result[0] += -0.01157669498705765;
                  } else {
                    result[0] += -0.06128208817588178;
                  }
                } else {
                  result[0] += 0.02509779585716239;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.06538847005160096;
              } else {
                result[0] += -0.06496357660367541;
              }
            } else {
              result[0] += 0.004978979857569205;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
              result[0] += -0.0181403127586867;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                    result[0] += -0.009532872489498858;
                  } else {
                    result[0] += 0.06147091390443731;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                        result[0] += -0.03532751595956437;
                      } else {
                        result[0] += 0.07507601259022678;
                      }
                    } else {
                      result[0] += 0.0090480290879362;
                    }
                  } else {
                    result[0] += -0.013409513819267855;
                  }
                }
              } else {
                result[0] += -0.029724165409256022;
              }
            }
          } else {
            result[0] += -0.029170067836123643;
          }
        } else {
          result[0] += -0.0011854945863710039;
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.0012795705072437703;
          } else {
            result[0] += -0.11570753411812562;
          }
        } else {
          result[0] += -0.018266220924528644;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
              result[0] += 0.022370734576124537;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.0024616305443272415;
                } else {
                  result[0] += -0.021477101475109034;
                }
              } else {
                result[0] += 0.020069878441877034;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += -0.04591706379422095;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.471622467041016513) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.55489373207092374) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.88192772865295499) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.004745860905819681;
                    } else {
                      result[0] += -0.05630727458443553;
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.0027596743760492093;
                    } else {
                      result[0] += -0.029179694028497627;
                    }
                  }
                } else {
                  result[0] += 0.04092477768693979;
                }
              } else {
                result[0] += 0.02324412413690423;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0011147868127397754;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.453179836273194248) ) ) {
                result[0] += 0.017700777191492253;
              } else {
                result[0] += -0.045714951512811804;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.04808697466726883;
                  } else {
                    result[0] += 0.024032223577254524;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                      result[0] += -0.030112109700773278;
                    } else {
                      result[0] += 0.032835254008801315;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.06415323034790674;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.022727636434187654;
                        } else {
                          result[0] += 0.06453361604000568;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.0072140391033330906;
                          } else {
                            result[0] += -0.14125699612807066;
                          }
                        } else {
                          result[0] += 0.046820516291303356;
                        }
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.09154971796329381;
                        } else {
                          result[0] += -0.12966459511066308;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0005726106520858058;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      result[0] += -0.006135427282654168;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
        result[0] += -0.0029262217472084866;
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += -0.0518727714714058;
        } else {
          result[0] += -0.019211077033132407;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.029427851768379417;
          } else {
            result[0] += 0.016616360825296817;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += 0.01392749050205766;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0007857733772164066;
                  } else {
                    result[0] += -0.04659563110070949;
                  }
                } else {
                  result[0] += 0.0020502543981449264;
                }
              } else {
                result[0] += 0.03712932875822311;
              }
            }
          } else {
            result[0] += -0.04435496918790268;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
          result[0] += 0.005509954331811293;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.05334142586493832;
          } else {
            result[0] += -0.00633608095512363;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.512487888336182529) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0429826739037719;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0002639784626156456;
                      } else {
                        result[0] += 0.03525008849475775;
                      }
                    }
                  } else {
                    result[0] += -0.001777433989855931;
                  }
                } else {
                  result[0] += 0.007530628615758534;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  result[0] += 0.011350448578054513;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.06281637858221216;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.384246587753296343) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.05318861535743982;
                      } else {
                        result[0] += -0.014482520677597602;
                      }
                    } else {
                      result[0] += 0.04521908131238013;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.003078998460995375;
              } else {
                result[0] += -0.04486190360648164;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06801711463348505;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.007402594204667622;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                    result[0] += -0.06154823141510696;
                  } else {
                    result[0] += 0.00570771998080222;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.020083321740264247;
                } else {
                  result[0] += -0.07168716759447756;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.015203487741676507;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.005717599971463237;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.0010159932257797757;
                        } else {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
                              result[0] += -0.05637480993793586;
                            } else {
                              result[0] += -0.006500285904665575;
                            }
                          } else {
                            result[0] += -0.0965172329306113;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.0008837144501386316;
                      } else {
                        result[0] += 0.0443149965191053;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.09450552334552442;
                        } else {
                          result[0] += 0.02044719182782874;
                        }
                      } else {
                        result[0] += -0.03266510012607274;
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.448852539062500444) ) ) {
                        result[0] += 0.03616407866240256;
                      } else {
                        result[0] += -0.05319903310881368;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.025471370007770906;
            } else {
              result[0] += 0.04177692695265578;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                result[0] += -0.06586191066531842;
              } else {
                result[0] += 0.0031845919719230907;
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.02413318739921001;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.13705248858557217;
                  } else {
                    result[0] += 0.022695406931506124;
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.0565858902447564;
                  } else {
                    result[0] += -0.05240072190988333;
                  }
                } else {
                  result[0] += -0.12525697672293257;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += 0.004927584649817404;
          } else {
            result[0] += -0.02808907086798001;
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.017030925556593714;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.07859009788602282;
              } else {
                result[0] += -0.02483906557221368;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += -0.01236943385760565;
              } else {
                result[0] += 0.03760807382015233;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          result[0] += -0.004864497589688385;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.010976086231993138;
          } else {
            result[0] += -0.04362876024800557;
          }
        }
      } else {
        result[0] += -0.052396335003174505;
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
          result[0] += 0.021262250454475533;
        } else {
          result[0] += -0.003776609094655589;
        }
      } else {
        result[0] += -0.019517661391964678;
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.075335502624512607) ) ) {
        result[0] += 0.0008936463703450282;
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            result[0] += -0.030620671085002688;
          } else {
            result[0] += 0.0031685881268874853;
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.058421781910292674;
          } else {
            result[0] += -0.008532310077055665;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.561026811599732333) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                      result[0] += 0.02613058745031885;
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.232009172439576083) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.689592361450196201) ) ) {
                          result[0] += 0.007588158732068736;
                        } else {
                          result[0] += 0.02450699697933154;
                        }
                      } else {
                        result[0] += -0.0108035146501355;
                      }
                    }
                  } else {
                    result[0] += 0.028522821516125247;
                  }
                } else {
                  result[0] += -0.0036853641534141386;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                  result[0] += 0.0069995040684088464;
                } else {
                  result[0] += -0.02609982641220894;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                  result[0] += 0.0037215058800028587;
                } else {
                  result[0] += -0.037876794285427895;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.01043333015700099;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.0542902939889554;
                    } else {
                      result[0] += -0.012936708107162485;
                    }
                  }
                } else {
                  result[0] += 0.015409902800127245;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.592359304428101474) ) ) {
                result[0] += -0.051283898257894284;
              } else {
                result[0] += -0.021457133281596277;
              }
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.010380380684946332;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
                  result[0] += -0.0027078057920115087;
                } else {
                  result[0] += -0.04224904081540646;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.09656204280746418;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.561026811599732333) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.795426130294800249) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.004509798633539841;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.0037327063344877102;
                      } else {
                        result[0] += -0.08067363869934045;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
                          result[0] += -0.0299693957966361;
                        } else {
                          result[0] += -0.001524979066055244;
                        }
                      } else {
                        result[0] += -0.06025662063912245;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.03860361607647633;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.054448582755335866;
                      } else {
                        result[0] += 0.017070030616711307;
                      }
                    } else {
                      result[0] += 0.004114081120067766;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  result[0] += -0.03706866547137663;
                } else {
                  result[0] += 0.01912144685518039;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0611029004360119;
                } else {
                  result[0] += -0.016288977873822396;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.05668041121342983;
                  } else {
                    result[0] += 0.019188802955627318;
                  }
                } else {
                  result[0] += 0.04072523236033553;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.584782838821412021) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.757321834564209873) ) ) {
                  result[0] += 0.036593256536025714;
                } else {
                  result[0] += 0.005931662998350202;
                }
              } else {
                result[0] += -0.0018654700379924915;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.029257680504984458;
              } else {
                result[0] += 0.008353566089861776;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.03909279584356755;
            } else {
              result[0] += 0.021544886837620306;
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.0749850919532588;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.00905922834106054;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.0883984565734881) ) ) {
                  result[0] += -0.09288416619870522;
                } else {
                  result[0] += 0.010834634570252695;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += -0.026100265749767093;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
                result[0] += -0.02578962475585474;
              } else {
                result[0] += 0.032128314474946525;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.010012074972730066;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
            result[0] += -0.027824092559390667;
          } else {
            result[0] += -0.05666066759170288;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.03332992944107023;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += -0.004687662568586189;
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.05219427607429055;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.014071295905551652;
                  } else {
                    result[0] += -0.08981465348434801;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.217645883560181552) ) ) {
                result[0] += -0.0971502833958216;
              } else {
                result[0] += 0.188070039544713;
              }
            } else {
              result[0] += -0.07476218094781033;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
          result[0] += 0.0264760098627291;
        } else {
          result[0] += 0.0017604754148052915;
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          result[0] += 0.013640959185805749;
        } else {
          result[0] += -0.03796880124657105;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      result[0] += -0.037401150300437426;
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.0016281399244690198;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.07252141595890568;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.12706515914873537;
                } else {
                  result[0] += -0.022805512518897567;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.023242253161962934;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
                    result[0] += 0.031964083339726466;
                  } else {
                    result[0] += 0.07198617705962232;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.019187040728108267;
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.11382474435164319;
                } else {
                  result[0] += 0.01934816291400249;
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.05596147963337666;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
                      result[0] += 0.027140082587332987;
                    } else {
                      result[0] += 0.10037632005085949;
                    }
                  }
                } else {
                  result[0] += -0.021178595802587213;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.044784551613554845;
            } else {
              result[0] += 0.012099944605151592;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              result[0] += -0.03311657366229579;
            } else {
              result[0] += -0.08142737498997571;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.02800295162123191;
                    } else {
                      result[0] += -0.10157464935191905;
                    }
                  } else {
                    result[0] += -0.01084445922664522;
                  }
                } else {
                  result[0] += 0.02645282382651621;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.007757302594387664;
                  } else {
                    result[0] += -0.03731291915082011;
                  }
                } else {
                  result[0] += 0.015331923322743754;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.01149121737615167;
                } else {
                  result[0] += -0.0865661077471768;
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.0029538809278509084;
                    } else {
                      result[0] += 0.052154053213627906;
                    }
                  } else {
                    result[0] += 0.08039950840926202;
                  }
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                      result[0] += -0.10561125820360594;
                    } else {
                      result[0] += 0.05767715914353362;
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                            result[0] += -0.13361429558818502;
                          } else {
                            result[0] += -0.005265685962902446;
                          }
                        } else {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.0042578558615349055;
                          } else {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                              result[0] += 0.017480360943335275;
                            } else {
                              result[0] += 0.0733636044126855;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.08052146983430991;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                        result[0] += -0.05530715184931321;
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                          result[0] += 0.09486825507007525;
                        } else {
                          result[0] += 0.02052377031156714;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.832297801971436435) ) ) {
                result[0] += -0.03299237778253174;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                  result[0] += -0.04870891411777075;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.07927829451117446;
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0442451943750659;
                    } else {
                      result[0] += -0.11433298157583999;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.09489547327128679;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.004569264826035494;
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
          if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += -0.04291387476661378;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.019896499755676016;
              } else {
                result[0] += 0.007495040203675038;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                result[0] += -0.00183970341570607;
              } else {
                result[0] += -0.044168205304888436;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.03756126997988098;
          } else {
            result[0] += -0.08745534082500911;
          }
        }
      }
    } else {
      result[0] += 0.004048810622347438;
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.453179836273194248) ) ) {
        result[0] += 0.0023760941701278927;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.0959255390868423;
          } else {
            result[0] += 0.009301747828602614;
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            result[0] += -0.04186679740008774;
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.004554121238646767;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.04888075473098805;
              } else {
                result[0] += -0.0043376641180478985;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.01314724774314999;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06871755409901241;
              } else {
                result[0] += -0.013392810388134085;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.161602735519410068) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.01989458930888123;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += -0.0255532089583502;
                } else {
                  result[0] += -0.06592981215313155;
                }
              }
            } else {
              result[0] += 0.00928676711282393;
            }
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.561026811599732333) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += 0.011552195445989892;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35526132583618342) ) ) {
                    result[0] += 0.008302003883643128;
                  } else {
                    result[0] += -0.0324384611692302;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.09110273635425277;
                  } else {
                    result[0] += -0.02663375523051579;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.013506352122798194;
                    } else {
                      result[0] += -0.04166605756790664;
                    }
                  } else {
                    result[0] += 0.007707678516077675;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.028568966747636135;
              } else {
                result[0] += 0.004540310509466394;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.176905632019043857) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                result[0] += -0.020526944282503918;
              } else {
                if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.03587892067487814;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.05503846886348834;
                      } else {
                        result[0] += 0.0026885296935626993;
                      }
                    } else {
                      result[0] += 0.020821640536812708;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.09929043767836801;
                    } else {
                      result[0] += -0.008665650372911893;
                    }
                  } else {
                    result[0] += 0.014467862321223655;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.04774548296035619;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.048184119425561646;
                    } else {
                      result[0] += 0.0014259137022591198;
                    }
                  } else {
                    result[0] += 0.04988752281814188;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.1073379085582134;
                } else {
                  result[0] += 0.03311260577357234;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.04302483701328974;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.043882284873996334;
              } else {
                result[0] += -0.00017802564196090507;
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                  result[0] += 0.026674911774150597;
                } else {
                  result[0] += -0.01579499762285735;
                }
              } else {
                result[0] += -0.049178887900843404;
              }
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += 0.029736133854357735;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.04582803359643063;
                } else {
                  result[0] += 0.0024643489303442833;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.7619357109069842) ) ) {
                result[0] += -0.07331391994986818;
              } else {
                result[0] += -0.006776743340814954;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.00625647799574222;
              } else {
                result[0] += -0.08642484241263293;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += -0.02479885182678881;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
                result[0] += -0.02466228730045822;
              } else {
                result[0] += 0.027710240678628612;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.009652342558901081;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.022488769128464963;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.016214387221323328;
              } else {
                result[0] += 0.15330675964861204;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.0015607962241485944;
              } else {
                result[0] += -0.06737215364313366;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.044418018993368086;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.012476859580852935;
              } else {
                result[0] += -0.03152967313967853;
              }
            } else {
              result[0] += -0.058761455470584915;
            }
          }
        }
      }
    } else {
      result[0] += 0.002985200058158612;
    }
  } else {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.329718828201294833) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += -0.001994478065082037;
                } else {
                  result[0] += -0.04495530218818925;
                }
              } else {
                result[0] += 0.012830966689186894;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.020127415657043901) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += 0.023270898852128734;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.042257966306547434;
                        } else {
                          result[0] += 0.011489114182476688;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.20949268341064631) ) ) {
                            result[0] += -0.003064425333570193;
                          } else {
                            result[0] += -0.05445183712277483;
                          }
                        } else {
                          result[0] += -0.05922231290021989;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.03657035771896327;
                      } else {
                        result[0] += 0.00489793957134502;
                      }
                    } else {
                      result[0] += 0.019783747315389233;
                    }
                  }
                } else {
                  result[0] += 0.04262486367220267;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                  result[0] += 0.006129335040956874;
                } else {
                  result[0] += -0.05900321573687272;
                }
              }
            }
          } else {
            result[0] += -0.030696770137067998;
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.0015918388982292383;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                result[0] += -0.01890393184898679;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += -0.009695446675431091;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.06170719606912492;
                    } else {
                      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.03217123172280309;
                        } else {
                          result[0] += -0.051706150219418524;
                        }
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.011567558935142492;
                        } else {
                          result[0] += 0.034048854017435015;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.028553388111971507;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
                          result[0] += -0.05387135799734462;
                        } else {
                          result[0] += 0.0072534365159542275;
                        }
                      } else {
                        result[0] += 0.02487207491867547;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                      result[0] += 0.03593016339865902;
                    } else {
                      result[0] += -0.0062250837845933375;
                    }
                  } else {
                    result[0] += -0.041183620330931;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
                        result[0] += 0.023253931971722555;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                          result[0] += 0.041932131494002926;
                        } else {
                          result[0] += -0.03197016703339009;
                        }
                      }
                    } else {
                      result[0] += -0.05163318460449573;
                    }
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.027097758392749943;
                        } else {
                          result[0] += -0.06037716425323524;
                        }
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
                          result[0] += 0.03978801906129227;
                        } else {
                          result[0] += -0.04790591582203629;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += -0.04231061765741281;
                      } else {
                        result[0] += 0.0015769792991879449;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.02298748040785405;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                      result[0] += -0.020696749025765383;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.01521682055026074;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.04149480722573578;
                          } else {
                            result[0] += 0.06310980129851727;
                          }
                        }
                      } else {
                        result[0] += -0.05295725375387198;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.06801952163381265;
                  } else {
                    result[0] += -0.011719009341000099;
                  }
                }
              }
            } else {
              result[0] += -0.04809570892129372;
            }
          }
        }
      } else {
        result[0] += 0.056010616712620004;
      }
    } else {
      result[0] += -0.061672533571361154;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY(  (data[32].missing != -1) && (data[32].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.03721183565086009;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += -0.04491255082478395;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.00011406003884933686;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.016419839906739977;
              } else {
                result[0] += -0.07667769456249723;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.087577104568482333) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.062074769150329584;
              } else {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.017214345055394247;
                } else {
                  result[0] += 0.02688410005997773;
                }
              }
            } else {
              result[0] += -0.00446700587631864;
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                result[0] += 0.024913915827056467;
              } else {
                result[0] += -0.06708488613389983;
              }
            } else {
              result[0] += -0.0012244364681603089;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
        result[0] += -0.007288187614773048;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
          result[0] += -0.026781362266813255;
        } else {
          result[0] += -0.05646034758374155;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.594409704208374912) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.860674262046814409) ) ) {
                result[0] += -0.0007688534340747006;
              } else {
                result[0] += -0.025786798066406477;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.551017761230469638) ) ) {
                result[0] += 0.0020871328765749075;
              } else {
                result[0] += -0.03975464204636364;
              }
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.00942563294278803;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.726826429367066318) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.025752561942598475;
                } else {
                  result[0] += 0.001137080361927051;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.010720571526125433;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.004363419685174656;
                  } else {
                    result[0] += 0.027180102244054767;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.0035340408642516017;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                  result[0] += 0.0035526547095041106;
                } else {
                  result[0] += 0.023874404138147298;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
                      result[0] += 0.021917295515490466;
                    } else {
                      result[0] += -0.03313040633179378;
                    }
                  } else {
                    result[0] += 0.04418959599211822;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                    result[0] += -0.010770988642890783;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                        result[0] += -0.039600815189604927;
                      } else {
                        result[0] += 0.008913393650780978;
                      }
                    } else {
                      result[0] += 0.027201261964863357;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.03748945900833455;
                } else {
                  result[0] += -0.03771996406117169;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.07604628384079076;
                } else {
                  result[0] += 0.047851887024469304;
                }
              } else {
                result[0] += -0.057180504076598665;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.005015711279009884;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += -0.004008778761417431;
                    } else {
                      result[0] += -0.08162984228828732;
                    }
                  }
                } else {
                  result[0] += -0.07480563381009915;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.020837402610529687;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.03331548576086295;
                    } else {
                      result[0] += -0.0013955248326561406;
                    }
                  } else {
                    result[0] += 0.04851637424873567;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
            result[0] += -0.03336594121111261;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.023271578363432206;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.18732333183288663) ) ) {
                  result[0] += -0.014280133871076812;
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.1555250292777609;
                      } else {
                        result[0] += 0.008271381994806716;
                      }
                    } else {
                      result[0] += 0.031319831239796085;
                    }
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += 0.08314099188861342;
                    } else {
                      result[0] += -0.09874283371763476;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.05804807079763703;
            }
          }
        } else {
          result[0] += -0.027445178829563793;
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
          result[0] += 0.002401731036347771;
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.000890401252254682;
          } else {
            result[0] += 0.01199125856422158;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.005858582640030696;
        } else {
          result[0] += 0.002504236910627616;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.009073197021085186;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.012962423027173847;
            } else {
              result[0] += -0.052560548873936075;
            }
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.00012329049896584753;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                result[0] += -0.004009978473713093;
              } else {
                result[0] += -0.04172678096712463;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += -0.014497663607584775;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.055608792852092914;
            } else {
              result[0] += -0.028115568397292734;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
          result[0] += 0.013029291982188433;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
              result[0] += 0.03015387120050382;
            } else {
              result[0] += -0.054947692169019346;
            }
          } else {
            result[0] += 0.004758701502640867;
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          result[0] += -0.037326188573214264;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.453179836273194248) ) ) {
            result[0] += -0.02392903492780263;
          } else {
            result[0] += 0.04145320737226435;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      result[0] += -0.0351236425759698;
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.0025926614139051497;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.109245061874390537) ) ) {
                result[0] += 0.001250508601695554;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                    result[0] += 0.019283987749907573;
                  } else {
                    result[0] += -0.05043440619195914;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.015685539165786636;
                    } else {
                      result[0] += -0.04856224700216427;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                        result[0] += -0.052271810368599195;
                      } else {
                        result[0] += -0.006265516067582686;
                      }
                    } else {
                      result[0] += 0.002518106518713443;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += 0.10477397391308561;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += -0.0728907652140552;
                  } else {
                    result[0] += 0.045131261518682374;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.03944423931635356;
                } else {
                  result[0] += 0.022158227341467548;
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.561026811599732333) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.09498887581462716;
                  } else {
                    result[0] += -0.019276532036763258;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0197798541839483;
                  } else {
                    result[0] += -0.0133636521075854;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.06807898572438378;
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.03707682318384013;
                      } else {
                        result[0] += -0.11958948444500675;
                      }
                    } else {
                      result[0] += 0.04303293444965808;
                    }
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.003107984829780688;
                    } else {
                      result[0] += 0.08031112899946623;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.901921629905701128) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0004765444013654457;
                } else {
                  result[0] += 0.05434215317283017;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += -0.05423485895859209;
                } else {
                  result[0] += 0.005857087986350077;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += -0.02389347188173626;
                } else {
                  result[0] += -0.06699855032865976;
                }
              } else {
                result[0] += -0.0677924897201476;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.00211572647094904) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.019130503591145324;
                } else {
                  result[0] += 0.001944870530940823;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.016619560972776986;
                } else {
                  if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.11508607259474363;
                  } else {
                    result[0] += 0.026957015306491014;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.011601635934155604;
                } else {
                  result[0] += -0.08258465483265152;
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.03931641308409903;
                  } else {
                    result[0] += 0.08777701677365657;
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.03844386342024175;
                  } else {
                    if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.06513810218114792;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.007242700247315408;
                        } else {
                          result[0] += 0.027070157594887612;
                        }
                      } else {
                        result[0] += -0.09425104930241063;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.052477706008833364;
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.035212713114651545;
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.06021905055187341;
            } else {
              result[0] += -0.017486130299335875;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                      result[0] += 0.0032323151058844066;
                    } else {
                      result[0] += -0.03453045623647199;
                    }
                  } else {
                    result[0] += 0.021223236252864902;
                  }
                } else {
                  result[0] += -0.028733620247474972;
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0136529385589612;
                } else {
                  result[0] += -0.06522226428804782;
                }
              }
            } else {
              result[0] += -0.07824131284969504;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.037699433255572075;
            } else {
              result[0] += -0.006288583375470814;
            }
          } else {
            result[0] += -0.0011228483314232656;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.05927062389696278;
          } else {
            result[0] += -0.003469141534640972;
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.0009822403026817058;
          } else {
            result[0] += -0.03777161342695545;
          }
        }
      } else {
        result[0] += -0.054136848661833485;
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      result[0] += -0.03313577751662913;
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.594409704208374912) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.774904012680054599) ) ) {
                result[0] += -0.0023235692586615854;
              } else {
                result[0] += -0.028812700402686714;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.551017761230469638) ) ) {
                result[0] += 0.001604653213008552;
              } else {
                result[0] += -0.037537992210280986;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.003525775737694574;
              } else {
                result[0] += -0.015509300699119839;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.0058216009381460605;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
                  result[0] += -0.031721849010954144;
                } else {
                  result[0] += 0.01628841748960882;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
              result[0] += 0.01690344908779059;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.32014131546020685) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
                      result[0] += 0.015025900523521526;
                    } else {
                      result[0] += -0.03175695376097436;
                    }
                  } else {
                    result[0] += 0.03368107477823468;
                  }
                } else {
                  result[0] += 0.0017878289638175848;
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.037586251546447506;
                } else {
                  result[0] += -0.028710043676289515;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.07801291845959181;
                } else {
                  result[0] += 0.038172225653127924;
                }
              } else {
                result[0] += -0.05575555851315298;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                    result[0] += -0.03923654774921498;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
                      result[0] += 0.0002476606587820234;
                    } else {
                      result[0] += 0.037016599984586;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += -0.0091203088156801;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.012113923545591496;
                    } else {
                      result[0] += -0.09178982629730129;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0047905273805113;
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.48872375488281428) ) ) {
                        result[0] += 0.01331348986413149;
                      } else {
                        result[0] += 0.11300617038518142;
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.10201846384134683;
                      } else {
                        result[0] += 0.011715718501919799;
                      }
                    }
                  } else {
                    result[0] += 0.052197262419783445;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.004388471529700492;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
              result[0] += 0.0029262933748955625;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += 0.03282577271165484;
              } else {
                result[0] += -0.051202315623011765;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.02757803433202004;
                } else {
                  result[0] += -0.013487869292352113;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0427739403623532;
                } else {
                  result[0] += -0.008170194535060665;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.018596668406159297;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += 0.05739981791907505;
                  } else {
                    result[0] += -0.014461116858995319;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += -0.04132477777628829;
                  } else {
                    result[0] += 0.021315388709538252;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY(  (data[31].missing != -1) && (data[31].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.03382067959096262;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
              result[0] += -0.053405882601488035;
            } else {
              result[0] += 0.024081511663179206;
            }
          } else {
            result[0] += -0.007456556265513114;
          }
        } else {
          result[0] += 0.0020588876742387526;
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.333273410797120029) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.03223633795814652;
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.0031498235749852217;
          } else {
            result[0] += -0.035248422597410026;
          }
        }
      } else {
        result[0] += -0.05479871310714007;
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        result[0] += 0.01399413462421203;
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.615975379943848544) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.034627460939605566;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.725620865821838823) ) ) {
              result[0] += 0.010348819973592834;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                    result[0] += -0.023038118198178317;
                  } else {
                    result[0] += 0.012588587179684505;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.034676176574687;
                          } else {
                            result[0] += -0.01777207934531019;
                          }
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.003064901056327392;
                          } else {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
                              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                                result[0] += 0.01308977651277197;
                              } else {
                                result[0] += 0.06384604815643287;
                              }
                            } else {
                              result[0] += 0.07135715523960172;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                          result[0] += 0.014722545727115472;
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                            result[0] += -0.07207838905464506;
                          } else {
                            result[0] += -0.012017392604212798;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.0392448170909363;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.04984081655287928;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                          result[0] += 0.025819375796823082;
                        } else {
                          result[0] += -0.0620880124957921;
                        }
                      }
                    } else {
                      result[0] += 0.038520604839606454;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.09392848777012619;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.03517420983844784;
                  } else {
                    result[0] += 0.020850073415435058;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            result[0] += 0.012224209950484904;
          } else {
            result[0] += -0.0316340644073961;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.002901427563777514;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.463808774948121005) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.045464822335102274;
                  } else {
                    result[0] += -0.05381295461688312;
                  }
                } else {
                  result[0] += -0.02659373226670303;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                    result[0] += -0.018426792252542592;
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.06243423978431004;
                      } else {
                        result[0] += 0.004832452352514616;
                      }
                    } else {
                      result[0] += -0.009433590313671901;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += 0.032466341972920296;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                        result[0] += 0.013678555864960175;
                      } else {
                        result[0] += -0.04857005771845804;
                      }
                    } else {
                      result[0] += -0.07222141599264319;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.057208127079767794;
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0016472505400868234;
                    } else {
                      result[0] += -0.1170079790647852;
                    }
                  } else {
                    result[0] += -0.09222888737577735;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                    result[0] += 0.05532589454620193;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06391651013413235;
                    } else {
                      result[0] += 0.0009676775363974266;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.047462744331170925;
                  } else {
                    result[0] += 0.005942331459761311;
                  }
                } else {
                  result[0] += -0.007983259790545448;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.015484769567628627;
                } else {
                  result[0] += -0.05552818776698574;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.019842048807736185;
                  } else {
                    result[0] += -0.005335229926617215;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += -0.07214787173709879;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.551017761230469638) ) ) {
                      result[0] += 0.021808938471348443;
                    } else {
                      result[0] += 0.06506299933709087;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.04775965228892297;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.022730452676216043;
      } else {
        if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.03297902937341247;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                      result[0] += -0.03843433897739706;
                    } else {
                      result[0] += 0.053737027012218955;
                    }
                  } else {
                    result[0] += -0.015440642870724414;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                    result[0] += 0.017226418548975733;
                  } else {
                    result[0] += -0.07952194570684118;
                  }
                }
              } else {
                result[0] += 0.021213516905328494;
              }
            } else {
              result[0] += -0.030996460413244272;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.66339445114135831) ) ) {
                result[0] += 0.029123945883271884;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.032283297737880835;
                  } else {
                    result[0] += -0.03736965829618706;
                  }
                } else {
                  result[0] += -0.03181509803248451;
                }
              }
            } else {
              result[0] += -0.04247932385292703;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.023700071310839853;
                } else {
                  result[0] += -0.026903700178400276;
                }
              } else {
                result[0] += -0.05578971843414964;
              }
            } else {
              result[0] += 0.011558416123413408;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.308072090148926669) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.00511107265838297;
        } else {
          result[0] += -0.021865045901766272;
        }
      } else {
        result[0] += -0.050425518087617964;
      }
    }
  } else {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.869292974472046787) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
            result[0] += 0.010841259800939605;
          } else {
            result[0] += -0.006804152381609673;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
            result[0] += 0.002557929667501254;
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.11326837539672896) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.05554325320719292;
              } else {
                result[0] += -0.010514833668407766;
              }
            } else {
              result[0] += 0.040974764462906293;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
          result[0] += -0.03309281411081468;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.010250632065504642;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.0033541001921565563;
                } else {
                  result[0] += -0.04426414808703626;
                }
              } else {
                result[0] += 0.003952277105347071;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.008206636507058848;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.035291729826834706;
                  } else {
                    result[0] += -0.01705687529519748;
                  }
                } else {
                  result[0] += 0.023934466956419;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.0013203527867252038;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                result[0] += -0.06345138441030652;
              } else {
                result[0] += -0.004477904592873817;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                result[0] += -0.007171473109283945;
              } else {
                result[0] += -0.054849121637282276;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                  result[0] += -0.036214152058927494;
                } else {
                  result[0] += 0.0035378846447555214;
                }
              } else {
                result[0] += 0.02924103635192983;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += -0.020155600470378024;
          } else {
            result[0] += -0.06539874488005941;
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.047926653469896524;
          } else {
            result[0] += -0.040093609820198134;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.020372889397822988;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.05328446026592759;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.84331607818603693) ) ) {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.0786701653744175;
                  } else {
                    result[0] += -0.002549379962904078;
                  }
                } else {
                  result[0] += 0.04964787733293668;
                }
              }
            }
          } else {
            if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.05452252435653774;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                  result[0] += -0.05533705354941874;
                } else {
                  result[0] += 0.04590952458379241;
                }
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.13422710052463302;
                } else {
                  result[0] += -0.011359347366894259;
                }
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.004226527290876389;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.004352210273290973;
                      } else {
                        result[0] += 0.04114001287068267;
                      }
                    } else {
                      result[0] += -0.07555353734615239;
                    }
                  }
                } else {
                  result[0] += -0.06598311233693065;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY(  (data[38].missing != -1) && (data[38].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
        result[0] += 0.04605933120629584;
      } else {
        result[0] += 0.0016529234906694727;
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.0527796373389261;
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.018328922803593828;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                result[0] += -0.04917815023284888;
              } else {
                result[0] += 0.09517023585127915;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += 0.007134592181888642;
                } else {
                  result[0] += -0.052453926694763335;
                }
              } else {
                result[0] += -0.08708561706200757;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.016285858679104055;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.025518857367164645;
              } else {
                if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += 0.005055949840991004;
                } else {
                  result[0] += -0.08279154177556458;
                }
              }
            }
          } else {
            result[0] += -0.03901460404324615;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
          result[0] += 0.010426742710972072;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
            result[0] += 0.008007899939000475;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.01084492373620556;
              } else {
                result[0] += -0.0738412529995092;
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.021843354731853604;
              } else {
                result[0] += -0.0637676397822744;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7631258964538592) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.99033999443054288) ) ) {
                result[0] += 0.0027768663708731076;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.03840766986003435;
                } else {
                  result[0] += -0.03528326598525164;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                result[0] += 0.04139783646167113;
              } else {
                result[0] += 0.00010769501206343748;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.05190089866054425;
            } else {
              result[0] += -0.012099436865672395;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.062734074238583;
          } else {
            result[0] += -0.01537360651244913;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.0009730928179586978;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.004061948409570202;
              } else {
                result[0] += -0.029699492000554824;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08823223481737773;
              } else {
                result[0] += -0.03990989652473864;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
              result[0] += -0.02423550846765881;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.017749688178959642;
              } else {
                result[0] += -0.021376796359168927;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.03734065542837133;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.018900932875934157;
            } else {
              result[0] += -0.02253722990064055;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.013325158541855273;
            } else {
              result[0] += -0.06823142433866708;
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.037501814502784524;
                } else {
                  result[0] += -0.028121274536048513;
                }
              } else {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.6149935722351092) ) ) {
                      result[0] += -0.020178089758736292;
                    } else {
                      result[0] += 0.014521694728949864;
                    }
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.11164686497487988;
                    } else {
                      result[0] += -0.026959012767908182;
                    }
                  }
                } else {
                  result[0] += 0.007802553970137981;
                }
              }
            } else {
              if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.03646391109771343;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.0014188031635402841;
                  } else {
                    result[0] += 0.016275330302530262;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.05576096952260349;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79285955429077326) ) ) {
                          result[0] += 0.008795394409798636;
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.014132204885671166;
                          } else {
                            result[0] += 0.0476386472850629;
                          }
                        }
                      } else {
                        result[0] += 0.06005162991575511;
                      }
                    } else {
                      result[0] += 0.0709107951879768;
                    }
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.006432036474125949;
                        } else {
                          result[0] += -0.13195410380230682;
                        }
                      } else {
                        result[0] += 0.02746465052537586;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += 0.0508087850351979;
                      } else {
                        result[0] += -0.11507828253796418;
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
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY(  (data[31].missing != -1) && (data[31].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.030621869544061042;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.054597061863547104;
            } else {
              result[0] += 0.005722846860566051;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.011622528531577206;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.0180989805314743;
              } else {
                result[0] += -0.027341498623907612;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
            result[0] += 0.006535131578580507;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.003964912305361333;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.03192940597427182;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += -0.02521314646502558;
                } else {
                  result[0] += 0.05243842011250213;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.308072090148926669) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.004493469161441722;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.03259702760723761;
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.07239213264745306;
            } else {
              result[0] += -0.014911184257248085;
            }
          }
        }
      } else {
        result[0] += -0.04821543768797934;
      }
    }
  } else {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
            result[0] += 0.00790779795044591;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.99033999443054288) ) ) {
                    result[0] += 0.003405916030416232;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.03517383852578137;
                    } else {
                      result[0] += -0.03344939410536379;
                    }
                  }
                } else {
                  result[0] += 0.021439864236336634;
                }
              } else {
                result[0] += -0.02370053103333325;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.056883298483806424;
              } else {
                result[0] += -0.01435979996408714;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.001860199496113722;
              } else {
                result[0] += -0.061029567320625616;
              }
            } else {
              result[0] += -0.005958860047646657;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.05719635410777005;
              } else {
                result[0] += -0.006462366505223114;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.002538664936529445;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.0017902164497853252;
                    } else {
                      result[0] += -0.04350211950015096;
                    }
                  } else {
                    result[0] += 0.016036875865413722;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                    result[0] += -0.01745417308681941;
                  } else {
                    result[0] += 0.018075519319505726;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.015976860306056866;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
                result[0] += 0.021046813847601833;
              } else {
                result[0] += 0.05432273371389371;
              }
            }
          } else {
            result[0] += -0.016470041693835456;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.561026811599732333) ) ) {
            result[0] += -0.00849702663254311;
          } else {
            result[0] += 0.01850532691487229;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
        result[0] += -0.03568314562232539;
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.09126859830154732;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              result[0] += -0.06619580114542065;
            } else {
              result[0] += -0.010390727038000551;
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.050901683307830764;
                  } else {
                    result[0] += -0.018101539535670466;
                  }
                } else {
                  result[0] += -0.1462904775317715;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.59645986557007014) ) ) {
                  result[0] += -0.11461517388485812;
                } else {
                  result[0] += 0.003319804247457959;
                }
              }
            } else {
              if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.05339157166192542;
              } else {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.13122549139045075;
                    } else {
                      result[0] += -0.022358582738998323;
                    }
                  } else {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.19066275304920854;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.017869498014292096;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11265373229980646) ) ) {
                          result[0] += 0.00580438034075583;
                        } else {
                          result[0] += 0.10240924849743627;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.039631287806531384;
                    } else {
                      result[0] += 0.017658856035884168;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += 0.045193616261161806;
                      } else {
                        result[0] += -0.07014275307071303;
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.01999781001787596;
                      } else {
                        result[0] += -0.12164821455978428;
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
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)10.00000000000000178) ) ) {
        result[0] += 0.02936679757324885;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0001149771537961412;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.06102277862725117;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
                        result[0] += -0.022825530833042576;
                      } else {
                        result[0] += 0.06883211282197074;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += 0.1378965342045104;
                        } else {
                          result[0] += -0.02335225488072212;
                        }
                      } else {
                        result[0] += -0.05461271893076654;
                      }
                    }
                  } else {
                    result[0] += 0.01703401498174203;
                  }
                }
              } else {
                result[0] += -0.029233912341182135;
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
              result[0] += -0.0503790124121982;
            } else {
              result[0] += 0.0029248495017325116;
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.138696432113648349) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
                  result[0] += -0.008725807521707821;
                } else {
                  result[0] += 0.029612146938496026;
                }
              } else {
                result[0] += -0.007224124269645193;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                  result[0] += 0.01865017531432968;
                } else {
                  result[0] += -0.0655890132825606;
                }
              } else {
                result[0] += -0.00322872235740867;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.10980303770840133;
              } else {
                result[0] += -0.04747626839803131;
              }
            } else {
              result[0] += 0.012065435742931842;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.24565792083740412) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.008213739317688594;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
              result[0] += -0.01831688343426108;
            } else {
              result[0] += -0.05231181139918283;
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.08496130438466097;
            } else {
              result[0] += -0.039260835091003365;
            }
          } else {
            result[0] += -0.026467132571226;
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.038763988641237174;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
              result[0] += -0.1315559570393773;
            } else {
              result[0] += 0.20165317437136254;
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.038608017483657574;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.002592421368025905;
              } else {
                result[0] += -0.06328801889673721;
              }
            } else {
              result[0] += 0.07873141954287595;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
              result[0] += 0.019460998300155713;
            } else {
              result[0] += -0.04340879638710992;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.561026811599732333) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.138696432113648349) ) ) {
                      result[0] += 0.005038358342418778;
                    } else {
                      result[0] += -0.015462209990380397;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.20949268341064631) ) ) {
                      result[0] += 0.007479191362576168;
                    } else {
                      result[0] += -0.014077547597558429;
                    }
                  }
                } else {
                  result[0] += -0.008915408824313232;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                    result[0] += 0.0007993692034871811;
                  } else {
                    result[0] += 0.015598361191930596;
                  }
                } else {
                  result[0] += -0.011028120820024745;
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.033560831851771486;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += -0.012302156208184383;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.009890912138286105;
                    } else {
                      result[0] += -0.07464328694300662;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.022482928431715787;
                      } else {
                        result[0] += -0.02889014975157595;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.158509254455567294) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                            result[0] += 0.008609747296726075;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.0883522033691424) ) ) {
                              result[0] += 0.024022406180206245;
                            } else {
                              result[0] += 0.07760116689853028;
                            }
                          }
                        } else {
                          result[0] += 0.043872051346254985;
                        }
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.925687789916993964) ) ) {
                            result[0] += -0.02067140958810684;
                          } else {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.08637642053731925;
                            } else {
                              result[0] += 0.0027551447908040426;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.01101738457753603;
                          } else {
                            result[0] += -0.08220387530038017;
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
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += 0.015179163679015843;
          } else {
            result[0] += -0.07510914852282245;
          }
        }
      } else {
        result[0] += 0.05077511280905949;
      }
    } else {
      result[0] += -0.05332218578874151;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY(  (data[45].missing != -1) && (data[45].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.028200346521012505;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
              result[0] += -0.043919349326727565;
            } else {
              result[0] += 0.01805474705840091;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.026312702856373577;
              } else {
                result[0] += -0.017492783963353336;
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.013569721965409004;
              } else {
                result[0] += -0.026493744047087625;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.99033999443054288) ) ) {
              result[0] += 0.010126879549532335;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                  result[0] += 0.016171341850396318;
                } else {
                  result[0] += -0.060674236911977555;
                }
              } else {
                result[0] += -0.003101140126312669;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.0958731552986799;
              } else {
                result[0] += -0.04595593964958444;
              }
            } else {
              result[0] += 0.007483446461997005;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.017026203467747313;
            } else {
              result[0] += 0.030680353922999518;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.17027091979980646) ) ) {
                result[0] += -0.06427144426098408;
              } else {
                result[0] += 0.15902834876577046;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.027834083975385893;
              } else {
                result[0] += 0.04315144282113301;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.17027091979980646) ) ) {
                result[0] += 0.013421274605693307;
              } else {
                result[0] += -0.16680412929403102;
              }
            } else {
              result[0] += -0.03823986926846276;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.09008352038093594;
            } else {
              result[0] += -0.004804458992833015;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.0004502370961711074;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.143469519850731;
            } else {
              result[0] += -0.05968150540307862;
            }
          } else {
            result[0] += -0.024788261746715955;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        result[0] += -0.004274118415794909;
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.03375712817918684;
              } else {
                result[0] += -0.013026185397899778;
              }
            } else {
              result[0] += 0.002215831158038184;
            }
          } else {
            result[0] += -0.027254788160846356;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.002194305945052005;
            } else {
              result[0] += -0.022954879283656953;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.00618069757395566;
                } else {
                  result[0] += -0.02680237490481714;
                }
              } else {
                result[0] += 0.0030301209135710676;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
                  result[0] += 0.006380703920173765;
                } else {
                  result[0] += -0.032885573663760694;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += -0.033315337082069615;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                        result[0] += -0.009240556031058385;
                      } else {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.05299012835595246;
                        } else {
                          result[0] += 0.012242983518315256;
                        }
                      }
                    } else {
                      result[0] += 0.028330856656923253;
                    }
                  }
                } else {
                  result[0] += -0.020326013305130623;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.767332553863526279) ) ) {
            result[0] += -0.009881653899351748;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.009216787672980188;
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                    result[0] += 0.01846272238093726;
                  } else {
                    result[0] += -0.08644252432923563;
                  }
                } else {
                  result[0] += 0.03635149408811874;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.138696432113648349) ) ) {
                  result[0] += -0.016122625779369733;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.014200451704064683;
                    } else {
                      result[0] += 0.04453276643196889;
                    }
                  } else {
                    result[0] += 0.06695745309449257;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.004849824516017325;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
              result[0] += 0.018567154114196936;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                result[0] += -0.04238362065529088;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.014852609563882818;
                } else {
                  result[0] += 0.023632752578669772;
                }
              }
            }
          } else {
            result[0] += -0.0084834905253947;
          }
        } else {
          result[0] += 0.0014957388725980192;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
          result[0] += 0.06654262805097393;
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.010298064732730892;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  result[0] += -0.002061151168135566;
                } else {
                  result[0] += 0.15106212902054267;
                }
              }
            } else {
              result[0] += -0.013277089898201472;
            }
          } else {
            result[0] += 0.06887669253998964;
          }
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
            result[0] += -0.015101513888255378;
          } else {
            result[0] += -0.04681572209529079;
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                result[0] += 0.019560352369112063;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.018893908569636395;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.07967154490479667;
                  } else {
                    result[0] += 0.0415122482245288;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                result[0] += -0.017595930853234517;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.04019025021642128;
                  } else {
                    result[0] += -0.028224601947899632;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.10260646172852797;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.13872336489028225;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.044917287723209104;
                      } else {
                        result[0] += -0.015353431287910405;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.0883522033691424) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.510617971420288974) ) ) {
                result[0] += -0.08041055506278705;
              } else {
                result[0] += 0.14812443332936825;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.06806838585929074;
              } else {
                result[0] += -0.02364181875318367;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.06526026762931414;
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.0005077105336260999;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
              result[0] += 0.017874059274176846;
            } else {
              result[0] += 0.04908833782045646;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.03666250602074154;
            } else {
              result[0] += 0.016430075097447486;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.490982532501221591) ) ) {
            result[0] += -0.014129561700657295;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.497191667556763583) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.033597603305450856;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += -0.019786785509898028;
                  } else {
                    result[0] += 0.04470302784644846;
                  }
                }
              } else {
                result[0] += 0.058166098438300874;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.039206423635095904;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += -0.0028211086261588935;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.529265403747559482) ) ) {
                    result[0] += 0.023607091138436176;
                  } else {
                    result[0] += 0.06803038084308285;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.09045838628787371;
              } else {
                result[0] += 0.050359444582433556;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                result[0] += -0.06671033177826745;
              } else {
                result[0] += -0.014253314639661209;
              }
            }
          } else {
            result[0] += -0.0685005570677225;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.055115990398418385;
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.025878440557460428;
              } else {
                result[0] += -0.0656307990824823;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.0012801639823354517;
                  } else {
                    result[0] += -0.08026498974109139;
                  }
                } else {
                  result[0] += 0.0029674745657544722;
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.016377194778821328;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06403121190081666;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.031175519342257586;
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.05430717631023707;
                      } else {
                        result[0] += -0.12213058196554888;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.13839721679687678) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                result[0] += -0.04543649642303583;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                  result[0] += -0.014220075266410493;
                } else {
                  result[0] += 0.024260020106240364;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                  result[0] += -0.017021213164965302;
                } else {
                  result[0] += -0.1265249079393217;
                }
              } else {
                result[0] += -0.010289637014853279;
              }
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.008484933904384047;
            } else {
              result[0] += 0.0508090583466139;
            }
          }
        } else {
          result[0] += 0.05000356047913865;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.02685782292197908;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.01069822392497495;
                  } else {
                    result[0] += -0.03113005889969976;
                  }
                } else {
                  result[0] += -0.05993347880167708;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += 0.0033171216517449776;
                } else {
                  result[0] += -0.03234883397787682;
                }
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.015604388539249604;
              } else {
                result[0] += -0.022204949471059127;
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += -0.011764240074544064;
            } else {
              result[0] += -0.05230068541066366;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                result[0] += 0.012226036362693857;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += 0.012968325692713632;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
                      result[0] += -0.010714837486554156;
                    } else {
                      result[0] += -0.0657484584731207;
                    }
                  }
                } else {
                  result[0] += 0.003995871442518736;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.08536327063350141;
                } else {
                  result[0] += -0.030119997188597432;
                }
              } else {
                result[0] += 0.01304883648859297;
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
                result[0] += -0.027515241455646544;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.002318936369048236;
                } else {
                  result[0] += 0.06359818307751869;
                }
              }
            } else {
              result[0] += -0.04168549808951439;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.196324348449708808) ) ) {
            result[0] += -0.026546483006920918;
          } else {
            result[0] += 0.019396498920431535;
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.02957375638771478;
              } else {
                result[0] += -0.08320719380369528;
              }
            } else {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.018145227524485892;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.06916798687093356;
                } else {
                  result[0] += 0.020531104774915588;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.848108768463135654) ) ) {
                result[0] += -0.007087282921186065;
              } else {
                result[0] += 0.06533134957331826;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                  result[0] += 0.017324247201066564;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += 0.007894076974729302;
                    } else {
                      result[0] += -0.054442458021452284;
                    }
                  } else {
                    result[0] += 0.02305201665682041;
                  }
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.47223544120788663) ) ) {
                  result[0] += -0.046085880966515676;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.10140766687192715;
                  } else {
                    result[0] += -0.020147728419496178;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.041903810549626;
          } else {
            result[0] += 0.014606173458387304;
          }
        } else {
          result[0] += -0.06793667090095559;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      result[0] += -0.02445774477248531;
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
              result[0] += 0.0021579302767050948;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                result[0] += -0.00035603698579550795;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.031103788728780196;
                } else {
                  result[0] += -0.002750397962725307;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.008340892076826235;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
                result[0] += -0.014199414385505635;
              } else {
                result[0] += 0.004555789387039042;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.047372142750677415;
                  } else {
                    result[0] += 0.015628154698490968;
                  }
                } else {
                  result[0] += -0.013808108678406414;
                }
              } else {
                result[0] += 0.004744477312462904;
              }
            } else {
              result[0] += -0.028965065217001346;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
              if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                result[0] += -0.002186436530501546;
              } else {
                result[0] += -0.019807156606647686;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.0014203270643322235;
                } else {
                  result[0] += -0.041566792820619466;
                }
              } else {
                result[0] += 0.006095623213015534;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.0033795917944238044;
          } else {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
              result[0] += 0.0509608931135125;
            } else {
              result[0] += -0.05289190021401013;
            }
          }
        } else {
          result[0] += -0.0011008176470537845;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
          result[0] += 0.007131734605437186;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.726826429367066318) ) ) {
            result[0] += 0.00017683867443252303;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.03845518645799765;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.929516315460205966) ) ) {
                  result[0] += -0.009991244440304288;
                } else {
                  result[0] += -0.05165198374642593;
                }
              } else {
                result[0] += -0.013453688010814003;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.02469179315476785;
        } else {
          result[0] += 0.0004053188645484764;
        }
      }
    } else {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += 0.0015784329153850866;
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
            result[0] += 0.014006636817519614;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += -0.01914333482117443;
            } else {
              result[0] += -0.0531660738257562;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.009960812780496521;
        } else {
          result[0] += 0.010923052515945114;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
          result[0] += 0.0020193354877814937;
        } else {
          result[0] += -0.012787105915780314;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.238486170768738237) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.002966022847387907;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                  result[0] += -0.022723519067192416;
                } else {
                  result[0] += 0.030182665075812853;
                }
              }
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += 0.09969045241386588;
              } else {
                result[0] += -0.05597181392093603;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
              result[0] += -0.039781473729454564;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.453179836273194248) ) ) {
                result[0] += -0.026976790566547962;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.680079460144043857) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.04778980294931779;
                    } else {
                      result[0] += 0.006505420562337917;
                    }
                  } else {
                    result[0] += -0.025076472178364423;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.010664360572984519;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.338555097579956943) ) ) {
                        result[0] += -0.10322705022429879;
                      } else {
                        result[0] += -0.02374966602702068;
                      }
                    }
                  } else {
                    result[0] += 0.04827434771603085;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.09996615122028875;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.10275312051818962;
              } else {
                result[0] += 0.013158701399581403;
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.310776710510254794) ) ) {
              result[0] += -0.04473834235154555;
            } else {
              result[0] += 0.003303016583832609;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += 0.009651106462943886;
            } else {
              result[0] += -0.014666533220318915;
            }
          } else {
            result[0] += -0.017766985586470895;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
                  result[0] += -0.004069437558784947;
                } else {
                  result[0] += -0.04193276729369687;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.05389684050436627;
                    } else {
                      result[0] += -0.009496513910063733;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += -0.04404212889057071;
                    } else {
                      result[0] += 0.02397604779916453;
                    }
                  }
                } else {
                  result[0] += 0.0384275382742325;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                result[0] += 0.0006081418096130524;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.005410835306138404;
                } else {
                  result[0] += 0.021560847193963294;
                }
              }
            }
          } else {
            result[0] += -0.010080039993454843;
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.04790919184849758;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.05507275313916315;
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.028826009772976044;
              } else {
                result[0] += -0.0020985339782134548;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
            result[0] += -0.03548860201577841;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.04716992838967905;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.013821494139510143;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0544676821175063;
                  } else {
                    result[0] += 0.02808914788193624;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                result[0] += 0.0016262994875176102;
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += 0.003928435024955651;
                } else {
                  if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.04932955910071717;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.02911440578834136;
                    } else {
                      result[0] += -0.08969149425549833;
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
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
            result[0] += 0.024154728413274637;
          } else {
            result[0] += -0.01720065862453313;
          }
        } else {
          result[0] += 0.0020580297460977824;
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.0009089487755231081;
        } else {
          result[0] += -0.02972259269140687;
        }
      }
    } else {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
              result[0] += -0.01760205303955946;
            } else {
              result[0] += -0.03827216893341795;
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.004042558899671657;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0060266428877218834;
                  } else {
                    result[0] += -0.061544862783778354;
                  }
                } else {
                  result[0] += -0.05646200516766514;
                }
              }
            } else {
              if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.004033697324695701;
                  } else {
                    result[0] += 0.029736953099843595;
                  }
                } else {
                  result[0] += -0.005316106248310217;
                }
              } else {
                result[0] += -0.015702081136993094;
              }
            }
          }
        } else {
          result[0] += 0.007422500616217563;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.021505048113909;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.026417016983033115) ) ) {
                  result[0] += -0.00012618389602324901;
                } else {
                  result[0] += 0.04327152906188692;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
                result[0] += -0.0004020687895209335;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                  result[0] += -0.01949187981941305;
                } else {
                  result[0] += 0.0769691160070406;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += -0.0020655708576858994;
                } else {
                  result[0] += -0.03803338076106742;
                }
              } else {
                result[0] += -0.059978434794669445;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.04657784874381632;
                  } else {
                    result[0] += -0.05150562267568203;
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.14540830340655247;
                  } else {
                    result[0] += -0.02151346467385127;
                  }
                }
              } else {
                result[0] += -0.015082434792798933;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.44381141662597834) ) ) {
            result[0] += 0.000532301638893549;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0041390700105764094;
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.01659585169457223;
              } else {
                result[0] += 0.0458532085881784;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.46093606948852717) ) ) {
          result[0] += -0.010170158372116163;
        } else {
          result[0] += 0.044925016536099295;
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          result[0] += 0.009274127160644973;
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.925687789916993964) ) ) {
                      result[0] += 0.018529705703200634;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                        result[0] += 0.010077021667323811;
                      } else {
                        result[0] += -0.024966446508614987;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                      result[0] += -0.015053533806925862;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.32868957519531428) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.01784071846509043;
                          } else {
                            result[0] += -0.03275926783002024;
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.603186130523683417) ) ) {
                            result[0] += -0.06007381548844916;
                          } else {
                            result[0] += -0.0029221441609914383;
                          }
                        }
                      } else {
                        result[0] += 0.02920509017897906;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                        result[0] += -0.010380447820273989;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                          result[0] += 0.016072943367811097;
                        } else {
                          result[0] += -0.04122191485906997;
                        }
                      }
                    } else {
                      result[0] += 0.02451493004768507;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
                      result[0] += 0.0072543448938551585;
                    } else {
                      result[0] += -0.015129448673111681;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += 0.0034865931825353957;
                  } else {
                    result[0] += -0.013542865376422498;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.17202329635620295) ) ) {
                      result[0] += 0.014450339087648263;
                    } else {
                      result[0] += -0.007308444598738176;
                    }
                  } else {
                    result[0] += -0.027850189274757242;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += 0.0018949277113379263;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.06297431778798511;
                  } else {
                    result[0] += -0.018350570550549058;
                  }
                } else {
                  result[0] += 0.20221823491273505;
                }
              }
            }
          } else {
            result[0] += 0.015347120870941417;
          }
        }
      }
    } else {
      result[0] += -0.001359962335290192;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY(  (data[30].missing != -1) && (data[30].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.025329525780713485;
      } else {
        result[0] += -0.00245287069304187;
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
            result[0] += -0.025854953232831185;
          } else {
            result[0] += 0.015164826801047974;
          }
        } else {
          result[0] += -0.01733865255930909;
        }
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.049284058933782925;
        } else {
          result[0] += -0.004885347451340925;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.869292974472046787) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70956039428711115) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.00638390664530569;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.035662601701680736;
                  } else {
                    result[0] += -0.005769428834739226;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.04837183872925133;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.499747991561890537) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.0030946496484570048;
                    } else {
                      result[0] += 0.044055489051477596;
                    }
                  } else {
                    result[0] += -0.017917078714970423;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                    result[0] += 0.007256480778569465;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.08808978234413534;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.932935476303101474) ) ) {
                            result[0] += 0.0045217067958243755;
                          } else {
                            result[0] += -0.0424592871770979;
                          }
                        } else {
                          result[0] += -0.0477470230659085;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += 0.031120535434780557;
                      } else {
                        result[0] += -0.03952338757148899;
                      }
                    }
                  }
                } else {
                  result[0] += 0.017446149693236702;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.04907117319195364;
                  } else {
                    result[0] += -0.009198278674695977;
                  }
                } else {
                  result[0] += 0.038262498645815685;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.063700929254523;
              } else {
                result[0] += -0.017388355023996178;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.052644031710707834;
                } else {
                  result[0] += -0.005061708764030441;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.004378219386025398;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.051159468427623894;
                    } else {
                      result[0] += -0.012499719005064774;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.0005171955053700289;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.06151224202314274;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                            result[0] += -0.03697086722749996;
                          } else {
                            result[0] += 0.008400416713647628;
                          }
                        } else {
                          result[0] += 0.021487342363706665;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.2692751884460467) ) ) {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.003728121558513995;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.463808774948121005) ) ) {
                          result[0] += -0.055054613254046286;
                        } else {
                          result[0] += -0.005792036562223924;
                        }
                      }
                    } else {
                      result[0] += 0.003256591292280641;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.0031201655233028458;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.046412715892256436;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                  if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                      result[0] += -0.055599745861738575;
                    } else {
                      result[0] += -0.005297508092425421;
                    }
                  } else {
                    result[0] += -0.07567504519730894;
                  }
                } else {
                  result[0] += 0.008578108823405152;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.07216628319173117;
                } else {
                  result[0] += 0.006742829611644219;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                  result[0] += -0.016732503668419746;
                } else {
                  result[0] += -0.05367902650546435;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                result[0] += -0.03803070653235254;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.874179124832154208) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.018359276865441053;
                      } else {
                        result[0] += -0.043939714344442084;
                      }
                    } else {
                      result[0] += 0.039004258682164755;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += 0.0024631313031115124;
                      } else {
                        result[0] += -0.08826680677160317;
                      }
                    } else {
                      result[0] += 0.030559296504634327;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.01824724163193729;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                      result[0] += -0.05704804249616835;
                    } else {
                      result[0] += 0.06318237051861883;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.04446590216503751;
      }
    } else {
      result[0] += -0.04324517086458329;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.026914532507591283;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY(  (data[38].missing != -1) && (data[38].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += -0.033903722629789836;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0013071324618546306;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.031483415602449646;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                    result[0] += 0.028153101652210757;
                  } else {
                    result[0] += -0.0091168503347738;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.1042793255921152;
                  } else {
                    result[0] += -0.018177657730252452;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.017492603994375543;
                  } else {
                    result[0] += -0.04644430303652133;
                  }
                } else {
                  result[0] += 0.03179063832975417;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.13030388981584456;
                      } else {
                        result[0] += 0.03088257538141438;
                      }
                    } else {
                      result[0] += -0.01890945155858465;
                    }
                  } else {
                    result[0] += -0.059209314769826375;
                  }
                } else {
                  result[0] += 0.0023700385339086825;
                }
              }
            } else {
              result[0] += 0.03157281291463362;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
              result[0] += -0.039030634719996335;
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.06047394053529292;
                } else {
                  result[0] += -0.0016576426152495353;
                }
              } else {
                result[0] += -0.01980501179115833;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.06257510185241788) ) ) {
              result[0] += -0.01522953390162305;
            } else {
              result[0] += 0.03573167401188442;
            }
          } else {
            result[0] += -0.026622428441201745;
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.061548676910487446;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0034973174969598324;
            } else {
              result[0] += -0.022347655665300212;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
          result[0] += -0.053757799163276124;
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.029566213711509026;
          } else {
            result[0] += 0.02007680440926659;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
        result[0] += 0.013502180209055062;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.018263728375184897;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.012854550693090437;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.11572060485854885;
              } else {
                result[0] += -0.0809130946502547;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.225547552108765537) ) ) {
            result[0] += -8.66336593918087e-05;
          } else {
            result[0] += -0.015575861355306576;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.0006815263480886287;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.08497194210109928;
          } else {
            result[0] += 0.0056856305124658715;
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0332970764136707;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                    result[0] += 0.03468607883168438;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                      result[0] += 0.010841369419686572;
                    } else {
                      result[0] += -0.07256153984774394;
                    }
                  }
                } else {
                  result[0] += -0.013999263653715142;
                }
              }
            } else {
              if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.012534658965135548;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.04378067704367825;
                    } else {
                      result[0] += 0.03151151530266824;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.05239424143411289;
                  } else {
                    result[0] += -0.0016086690050918187;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                  result[0] += 0.014706406199174949;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.05079782282329496;
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.08037775328812936;
                      } else {
                        result[0] += -0.0039315691658733795;
                      }
                    } else {
                      result[0] += 0.008656147425352307;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.02032613959912095;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.97887301445007413) ) ) {
                  result[0] += -0.005088447277984451;
                } else {
                  result[0] += 0.026381156263466272;
                }
              }
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.09400495630894341;
                } else {
                  result[0] += -0.03950920753298727;
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.010667866102230493;
                } else {
                  result[0] += -0.08053003804555847;
                }
              }
            }
          }
        } else {
          result[0] += -0.04766346266788141;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.02607548201059572;
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.03154403623101654;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.95005559921264826) ) ) {
                            result[0] += 0.005056127811673379;
                          } else {
                            result[0] += 0.08788378115993994;
                          }
                        } else {
                          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.011900981128992334;
                          } else {
                            result[0] += -0.039805558197667096;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                          result[0] += -0.08175710153907534;
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                            result[0] += 0.019420722654083584;
                          } else {
                            result[0] += 0.09043509062126664;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.021359995624087733;
                    }
                  } else {
                    result[0] += -0.024007440495815643;
                  }
                } else {
                  result[0] += 0.0111995782598007;
                }
              } else {
                result[0] += -0.02296230675633657;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.358708143234253818) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.138696432113648349) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.006162253825206563;
                      } else {
                        result[0] += -0.04578633758402014;
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                          result[0] += 0.05390684547253614;
                        } else {
                          result[0] += -0.030688893829217495;
                        }
                      } else {
                        result[0] += 0.025777895245135424;
                      }
                    }
                  } else {
                    result[0] += -0.0010570551285862008;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += 0.07176195386229128;
                    } else {
                      result[0] += -0.031050874428984684;
                    }
                  } else {
                    result[0] += 0.01070500157116905;
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                    result[0] += 0.00654589722383179;
                  } else {
                    result[0] += -0.04933034450566398;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                    result[0] += -0.04462338045492595;
                  } else {
                    result[0] += 0.020762778962051694;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
                result[0] += -0.035572554006317245;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += -0.054293290575398646;
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.007192649897820623;
                  } else {
                    result[0] += 0.040392477488413084;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.057677709800884935;
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.198464870452881303) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.0014383579194902942;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.010405896144986507;
                    } else {
                      result[0] += -0.06255187838542345;
                    }
                  }
                } else {
                  result[0] += 0.03058965359920105;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.05314092085908676;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.014670289919121837;
                  } else {
                    result[0] += -0.010757759192544049;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.08388417577225388;
              } else {
                result[0] += -0.07534932640749262;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.65810394287109553) ) ) {
              result[0] += -0.07301823445518939;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += -0.07680701251506758;
              } else {
                result[0] += 0.04307733154305651;
              }
            }
          }
        } else {
          result[0] += -0.03384713769535586;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += -0.07543463009907864;
          } else {
            result[0] += -0.033693727339031435;
          }
        } else {
          result[0] += -0.017672248762779855;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
          result[0] += 0.020084918178524598;
        } else {
          result[0] += -0.03825576014438221;
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.0010067847557048723;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                result[0] += 0.02585767277686543;
              } else {
                result[0] += -0.01549378834684922;
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.06471726206887583;
                } else {
                  result[0] += -0.007508100724147692;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                    result[0] += 0.017611895076615727;
                  } else {
                    result[0] += -0.049358771712245314;
                  }
                } else {
                  result[0] += 0.0022264424399550346;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.016115593939043178;
            } else {
              result[0] += -0.02556021490267732;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.019636283546758393;
            } else {
              result[0] += 0.004040657803148421;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
        result[0] += 0.026862420503793467;
      } else {
        result[0] += -0.0677014463977044;
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
      if ( UNLIKELY(  (data[45].missing != -1) && (data[45].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.025179596334421062;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.03069577701662467;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0015187320693899455;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03043679935929855;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.00705298498215797;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.0987492648868088;
                  } else {
                    result[0] += -0.013130151371711477;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                  result[0] += 0.031036377445512322;
                } else {
                  result[0] += 0.1114399946643478;
                }
              } else {
                result[0] += 0.014667766416073045;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.524927973747253862) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
                  result[0] += 0.008676489214292153;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
                    result[0] += -0.05101637234515578;
                  } else {
                    result[0] += -0.006479114680034585;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.53498554229736506) ) ) {
                  result[0] += 0.05802247629395645;
                } else {
                  result[0] += -0.020480261102259978;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                result[0] += -0.05190480168325549;
              } else {
                result[0] += 0.0025116817334379614;
              }
            } else {
              result[0] += -0.0329540236468222;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.645421981811524326) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.198464870452881303) ) ) {
            result[0] += -0.029854596629639796;
          } else {
            result[0] += 0.013250997652271897;
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.024482420560727833;
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0449058982503792;
            } else {
              result[0] += -0.009953671681547047;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.0018333705290133524;
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.02807724237854883;
          } else {
            result[0] += -0.06540085884006074;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.161602735519410068) ) ) {
          result[0] += 0.00801393202067551;
        } else {
          result[0] += -0.01225360912914107;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.015665298116327327;
                } else {
                  result[0] += -0.05548633586881997;
                }
              } else {
                if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.011530317178505337;
                } else {
                  result[0] += -0.027151477057316693;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.35312032699585139) ) ) {
                result[0] += 0.026051043888857657;
              } else {
                result[0] += -0.007023897806681775;
              }
            }
          } else {
            result[0] += -0.045323491736901866;
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.05160666249562629;
          } else {
            result[0] += -0.014083517240227675;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
          result[0] += -0.000504561173284667;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.00441763088391391;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.0025026605830428053;
                } else {
                  result[0] += -0.04233157540850665;
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                    result[0] += -0.04675775198820833;
                  } else {
                    result[0] += -0.00358755063901257;
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.08820097378246983;
                  } else {
                    result[0] += -0.04698885674704323;
                  }
                }
              }
            }
          } else {
            result[0] += 0.002974147476159485;
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.028335171970831276;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.016856261335942934;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.06456550310025;
              } else {
                result[0] += -0.0156765615810703;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += -0.033252547284632034;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.062311499776208534;
                  } else {
                    result[0] += -0.014853254396033294;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.015673264886554326;
                    } else {
                      result[0] += 0.044407477378575595;
                    }
                  } else {
                    result[0] += -0.015838440616726526;
                  }
                }
              } else {
                result[0] += -0.009642314213105966;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.00211572647094904) ) ) {
                      result[0] += -0.016187950944255677;
                    } else {
                      result[0] += 0.021258075119410823;
                    }
                  } else {
                    result[0] += -0.04924669940568429;
                  }
                } else {
                  result[0] += 0.03206890382785951;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                  result[0] += 0.026466700208966316;
                } else {
                  result[0] += 0.005069352786147617;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
        result[0] += 0.04811385666370942;
      } else {
        if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.06397258338187096;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
            result[0] += 0.084244058968023;
          } else {
            result[0] += 0.007031340619257954;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.042636274660403914;
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.701612949371339667) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.768316030502320224) ) ) {
                  result[0] += -0.004326678423342267;
                } else {
                  result[0] += -0.030909522358919073;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += -0.05218447396405739;
                } else {
                  result[0] += 0.006494094757225332;
                }
              }
            } else {
              result[0] += -0.003190794831414761;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                  result[0] += 0.007916655004456917;
                } else {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.05144371186843685;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.016605059300106396;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.012295316977212183;
                      } else {
                        result[0] += 0.07762789413069911;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                    result[0] += -0.0060775257795479395;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += -0.009610915184316867;
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.010222406061257966;
                        } else {
                          result[0] += 0.13482787437763963;
                        }
                      } else {
                        result[0] += 0.060897262654420074;
                      }
                    }
                  }
                } else {
                  result[0] += -0.029878202239646125;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.0883522033691424) ) ) {
                result[0] += -0.06759838902952181;
              } else {
                result[0] += 0.01626278986635978;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.113679647445679599) ) ) {
            result[0] += -0.0710806052908524;
          } else {
            result[0] += 0.03971493925793289;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.161602735519410068) ) ) {
          result[0] += 0.007234809020017548;
        } else {
          result[0] += -0.011606434047676504;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.0029522388548122985;
          } else {
            result[0] += -0.07921174873080028;
          }
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.05030503292390029;
            } else {
              result[0] += -0.0102681392846689;
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.014511913200241956;
            } else {
              result[0] += -0.014726928220698763;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.04259724848077581;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.002848847377565117;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.161602735519410068) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.003906740283209978;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
                  result[0] += -0.045567719323630074;
                } else {
                  result[0] += -0.0164880011011671;
                }
              }
            } else {
              result[0] += 0.01478256750909274;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.05560928795787395;
              } else {
                result[0] += -0.05148633777805953;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.05189541394626841;
                } else {
                  result[0] += 0.01380587905890552;
                }
              } else {
                result[0] += -0.0020448041088445877;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                result[0] += -0.004955813678860604;
              } else {
                result[0] += -0.046486433955849914;
              }
            } else {
              result[0] += -0.00850124879118521;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.012731970087811456;
            } else {
              result[0] += -0.05564036844701597;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.010253447665949891;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.0372293728331115;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.0006843811612349432;
                      } else {
                        result[0] += -0.04833971181170246;
                      }
                    } else {
                      result[0] += 0.014494561931976997;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.797939777374268466) ) ) {
                      result[0] += -0.017147303190157143;
                    } else {
                      result[0] += 0.01808320542031487;
                    }
                  } else {
                    result[0] += 0.011632898836002483;
                  }
                } else {
                  result[0] += 0.0211807904608438;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.000649397896605242;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.463808774948121005) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.013744780188036846;
                    } else {
                      result[0] += -0.07945192267697143;
                    }
                  } else {
                    result[0] += -0.005686185370358314;
                  }
                }
              } else {
                result[0] += 0.007193496876515168;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
            result[0] += -0.011417299392137394;
          } else {
            result[0] += 0.020614855206988573;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
            result[0] += 0.017235549516209762;
          } else {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06567235447594652;
            } else {
              result[0] += -0.013471773790652156;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += -0.036044142070911576;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.01890403737465738;
                } else {
                  result[0] += 0.0929340132799884;
                }
              } else {
                if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.0070221782112289055;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.0011439560657804469;
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01255543517429594;
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.018842486520393963;
                      } else {
                        result[0] += -0.058840629433888084;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.012134562225981992;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.11083764567199889;
                  } else {
                    result[0] += -0.010848475697247681;
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.02016600418521226;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.005545832469183229;
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.57691621780395685) ) ) {
                          result[0] += -0.07831360390239389;
                        } else {
                          result[0] += 0.09382264650649366;
                        }
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.009802659254214661;
                          } else {
                            result[0] += -0.08722081943862176;
                          }
                        } else {
                          result[0] += 0.11970460085862583;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.06733920320843888;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.02550293465148967;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06446285906561743;
                } else {
                  result[0] += -0.0008095238903154553;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.022908700890965375;
          } else {
            result[0] += -0.060156060869583305;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
            result[0] += 0.016900020303955245;
          } else {
            result[0] += 0.0612181483156034;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
              result[0] += 0.01120298685741984;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                        result[0] += 0.20753764441975614;
                      } else {
                        result[0] += 0.020734915011517294;
                      }
                    } else {
                      result[0] += 0.02200072394406644;
                    }
                  } else {
                    result[0] += -0.009620165559455782;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                    result[0] += -0.010306579660831126;
                  } else {
                    result[0] += -0.05926869239331844;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                    result[0] += -0.05978291817528995;
                  } else {
                    result[0] += -0.011048374517869185;
                  }
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    result[0] += 0.051865521802647196;
                  } else {
                    result[0] += 0.008022575381071284;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54296922683715998) ) ) {
              result[0] += 0.04505352155877661;
            } else {
              result[0] += -0.048366826729734747;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.04946641569579015;
        } else {
          result[0] += -0.0024302278452903586;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.0010084353272819607;
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.03176250368442562;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                result[0] += -0.01524703896725615;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.0034510926939508284;
                  } else {
                    result[0] += -0.05173623468067848;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.524927973747253862) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.566809177398682529) ) ) {
                      if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += 0.03042684486915197;
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                                result[0] += -0.02941666185742436;
                              } else {
                                result[0] += -0.12427024725620728;
                              }
                            } else {
                              result[0] += 0.05756737237166296;
                            }
                          } else {
                            result[0] += 0.011678809466947895;
                          }
                        } else {
                          result[0] += -0.08700702349056949;
                        }
                      }
                    } else {
                      result[0] += 0.04224098108997063;
                    }
                  } else {
                    result[0] += -0.0335471404003064;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.03743168576491391;
        }
      } else {
        result[0] += -0.040154430081336456;
      }
    } else {
      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
        result[0] += 0.01919436696431675;
      } else {
        result[0] += -0.06619595045714904;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81890821456909357) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.02378818308443675;
          } else {
            result[0] += 0.005163256458937623;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
            result[0] += 0.006898021972857109;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03290776598870585;
              } else {
                result[0] += -0.03954409743790886;
              }
            } else {
              result[0] += 0.011870648780911309;
            }
          }
        }
      } else {
        result[0] += -0.019926902968275456;
      }
    } else {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.333273410797120029) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.008362640277773793;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                        result[0] += -0.019462525896192955;
                      } else {
                        result[0] += 0.03948234548567759;
                      }
                    } else {
                      result[0] += -0.029707174614282475;
                    }
                  } else {
                    result[0] += -0.06362786074907696;
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += -0.024315821793008053;
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.006293513243201586;
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                            result[0] += 0.012180971049377114;
                          } else {
                            result[0] += -0.042907951643839974;
                          }
                        }
                      } else {
                        result[0] += 0.019259293412398574;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.10445635569518608;
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.145586729049683505) ) ) {
                        result[0] += -0.03714637241726495;
                      } else {
                        result[0] += 0.08567540218142788;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                      result[0] += -0.04253832010555243;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                        result[0] += -0.07273559449493455;
                      } else {
                        result[0] += 0.04722184804531188;
                      }
                    }
                  } else {
                    result[0] += -0.0020619679497268033;
                  }
                }
              }
            } else {
              result[0] += -0.03260653108723693;
            }
          } else {
            result[0] += -0.028320189894443133;
          }
        } else {
          result[0] += 0.007701033903925652;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.002355147406379085;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.026830611572505297;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.023247495303439523;
                } else {
                  result[0] += 0.06622072183791924;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.03627183222754265;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += 0.0024373293087676055;
                    } else {
                      result[0] += -0.055453009643875645;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
                      result[0] += -0.006183108454059067;
                    } else {
                      result[0] += -0.03942655800457624;
                    }
                  }
                } else {
                  result[0] += 0.028826457846901916;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.04407488373543669;
                  } else {
                    result[0] += -0.04640517041058504;
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.14069211900961162;
                  } else {
                    result[0] += -0.019406587195586987;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.02403595344430933;
                  } else {
                    result[0] += -0.13868253500764238;
                  }
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.12171949252541613;
                    } else {
                      result[0] += 0.009594863112012077;
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.08351346540752119;
                      } else {
                        result[0] += -0.02295338933759262;
                      }
                    } else {
                      result[0] += -0.019166189487199312;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
            result[0] += 0.0007018874534257784;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.005269091282995035;
            } else {
              result[0] += 0.0291780231729636;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.847910165786744052) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.27480554580688654) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.0018063760784189867;
              } else {
                result[0] += -0.020004593713110034;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.04342708222544557;
              } else {
                result[0] += -0.019472672820261896;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.018649180186559543;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.020737896918478764;
              } else {
                result[0] += -0.029258060569191863;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.006057702136193515;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
              result[0] += 0.01133780690800561;
            } else {
              result[0] += 0.04722728834585854;
            }
          }
        }
      } else {
        result[0] += 0.0036527794414566816;
      }
    } else {
      result[0] += -0.0011904001086077355;
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.026315014601635052;
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.001404816390604537;
        } else {
          result[0] += 0.03606065481444341;
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.021477342989314932;
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.04590762738006109;
              } else {
                result[0] += -0.10129967357718453;
              }
            }
          } else {
            result[0] += -0.008787232242675478;
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.00011844585992950427;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.012418974350865003;
            } else {
              result[0] += -0.07662477586293401;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                    result[0] += -0.0006673974169858479;
                  } else {
                    result[0] += 0.01520499697518446;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                    result[0] += 0.018161903676039672;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.01230932888715595;
                      } else {
                        result[0] += -0.05907298378401502;
                      }
                    } else {
                      result[0] += 0.01972727369342895;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06564815962790858;
                } else {
                  result[0] += 0.004770559413850416;
                }
              }
            } else {
              result[0] += 0.025086027666229263;
            }
          } else {
            result[0] += -0.0016789942158761827;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
            result[0] += 0.004040445550260821;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.011811552786848275;
              } else {
                result[0] += -0.07151294229953768;
              }
            } else {
              result[0] += 0.004158241814360994;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.006325109738734282;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                result[0] += 0.026870522642372704;
              } else {
                result[0] += -0.003660277659091599;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.04187270748410186;
            } else {
              result[0] += -0.00862706472040107;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += 0.07696633013116085;
          } else {
            result[0] += -0.054156245849240864;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.012847872262612824;
          } else {
            result[0] += -0.05669528982452901;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0018504011871480913;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.279299736022950107) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
                result[0] += -0.010985192659666639;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.01851818991136576;
                } else {
                  result[0] += -0.05668205429625416;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                result[0] += -0.032502474870134225;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.007940187092261141;
                } else {
                  result[0] += 0.045700395896909246;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.00023537187872064362;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.007630456896905044;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.026220431655852867;
                } else {
                  result[0] += -0.07186010704246307;
                }
              }
            }
          } else {
            result[0] += 0.002521217287691009;
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.02788908807948672;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                    result[0] += -0.003850563860906732;
                  } else {
                    result[0] += -0.04615806169062095;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.008958757236349094;
                  } else {
                    result[0] += 0.046135063915020154;
                  }
                }
              } else {
                result[0] += 0.030514874820819462;
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.02612396909476883;
                } else {
                  result[0] += 0.005673899602969021;
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.0123794392777212;
                    } else {
                      result[0] += -0.028329931399851228;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                      result[0] += 0.053984694507320075;
                    } else {
                      result[0] += 0.017406498160530817;
                    }
                  }
                } else {
                  result[0] += -0.033488521516797244;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.32868957519531428) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0042953350676375604;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += -0.015639316107589868;
                    } else {
                      result[0] += -0.08450771361137559;
                    }
                  } else {
                    result[0] += -0.011249802458907936;
                  }
                }
              } else {
                result[0] += 0.007117090935589847;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.42478513717651456) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.603186130523683417) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
            result[0] += 0.0031581635284406427;
          } else {
            result[0] += 0.015304497412389465;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.07851773914129119;
            } else {
              result[0] += -0.025029773522266353;
            }
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.00879228038778425;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.529265403747559482) ) ) {
                  result[0] += -0.008595926592190354;
                } else {
                  result[0] += -0.04195046840448695;
                }
              } else {
                result[0] += 0.012214079807203284;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
          result[0] += 0.01061503517409268;
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.08003254730610125;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                result[0] += 0.02916447582326729;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                    result[0] += -0.024436179022467638;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40695333480835139) ) ) {
                      result[0] += 0.01562647730368153;
                    } else {
                      result[0] += -0.03360275603004587;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                    result[0] += -0.004127986636554872;
                  } else {
                    result[0] += -0.04251088667500811;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.00211572647094904) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.011398857365636995;
                } else {
                  result[0] += -0.016422408257043018;
                }
              } else {
                result[0] += -0.018719817492773117;
              }
            } else {
              result[0] += 0.038880101796944365;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
        result[0] += 0.0041624429339508304;
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.028462612026609657;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.04408594826917006;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.009323911737760654;
                } else {
                  result[0] += -0.048955271854805756;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.097527027130127841) ) ) {
                  result[0] += -0.025148758806614838;
                } else {
                  result[0] += -0.06850678155627168;
                }
              }
            } else {
              result[0] += -0.0016397089733143353;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.05333444081406024;
        } else {
          result[0] += -0.0024157718863645618;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.966960191726685458) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
            result[0] += -0.020396839084629522;
          } else {
            result[0] += -0.05372612527533597;
          }
        } else {
          result[0] += -0.006570785166281033;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          result[0] += 0.04899962681614087;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.01602290514070396;
          } else {
            result[0] += -0.04993516792958461;
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.00740819966041989;
                  } else {
                    result[0] += -0.09079067205766428;
                  }
                } else {
                  result[0] += -0.10316670392531535;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                    result[0] += -0.056670314730259734;
                  } else {
                    result[0] += 0.03061792524862056;
                  }
                } else {
                  result[0] += 0.06934090111862122;
                }
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.004005304353284295;
                  } else {
                    result[0] += -0.031202226102959;
                  }
                } else {
                  result[0] += 0.004937497253833676;
                }
              } else {
                result[0] += 0.009463253094193802;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.00310463210775188;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.5655555725097674) ) ) {
                  result[0] += -0.032736127499494165;
                } else {
                  result[0] += -0.002617356299542358;
                }
              }
            } else {
              result[0] += -0.03482636675743871;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.007384120012138827;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.02653819005022955;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    result[0] += -0.009412062519074452;
                  } else {
                    result[0] += -0.05909912281712644;
                  }
                } else {
                  if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.027740227953036312;
                  } else {
                    result[0] += -0.00033495087771701467;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.0010312049709200767;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                    result[0] += -0.005794771231049871;
                  } else {
                    result[0] += 0.05916646858074318;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.04882380355202162;
                  } else {
                    result[0] += 0.010107190148277247;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                  result[0] += -0.04360871102494251;
                } else {
                  result[0] += 0.05028197810043614;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.499747991561890537) ) ) {
            result[0] += -0.00403363662857677;
          } else {
            result[0] += -0.034835047621652834;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
              result[0] += 0.00302835794484774;
            } else {
              result[0] += -0.029058226059328077;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
              result[0] += 0.026538396316957338;
            } else {
              result[0] += -0.02068264776920727;
            }
          }
        }
      } else {
        result[0] += -0.03686480481968143;
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.847910165786744052) ) ) {
            result[0] += -0.003408995269024637;
          } else {
            result[0] += -0.04199473842856538;
          }
        } else {
          result[0] += 0.0100347593577879;
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.022263696589225966;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.231051445007325107) ) ) {
            result[0] += -0.002696192557785363;
          } else {
            result[0] += -0.028472650721887788;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.170116901397705966) ) ) {
              result[0] += 0.012051179129815055;
            } else {
              result[0] += -0.023918597781807202;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += 0.001572165051819084;
            } else {
              result[0] += -0.03551678979897025;
            }
          }
        } else {
          result[0] += 0.023680821129161565;
        }
      } else {
        result[0] += -0.08116950018242954;
      }
    } else {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 2.910427112903841e-06;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.847910165786744052) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.011810110413147473;
              } else {
                result[0] += -0.05512159403828623;
              }
            } else {
              result[0] += -0.06661547521625981;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.039548465516316446;
                  } else {
                    result[0] += -0.053772582748557354;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.042395656084773414;
                  } else {
                    result[0] += 0.004936331840528084;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.85305833816528498) ) ) {
                      result[0] += -0.016056951510978367;
                    } else {
                      result[0] += 0.03712870795633676;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.05108466072093663;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05479049682617365) ) ) {
                        result[0] += -0.029544555769961758;
                      } else {
                        result[0] += 0.04115837048308864;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                        result[0] += -0.03752875533012924;
                      } else {
                        result[0] += 0.01908893423043724;
                      }
                    } else {
                      result[0] += 0.021683584757992935;
                    }
                  } else {
                    result[0] += -0.031811997211528185;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.06221215209653496;
                  } else {
                    result[0] += 0.0006906215114093057;
                  }
                } else {
                  result[0] += -0.06528766940946294;
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += 0.024667524592221792;
                } else {
                  result[0] += -0.04420344756053768;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += -0.0009095702305451288;
          } else {
            result[0] += -0.042011934592987264;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.0006464031516101264;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                result[0] += -0.05972452247673802;
              } else {
                result[0] += -0.006950241801197728;
              }
            }
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.001953499137678194;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01584624214810452;
                    } else {
                      result[0] += -0.028841008761429324;
                    }
                  } else {
                    result[0] += 0.016460642385527944;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.008209901952910435;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += 0.028522505138153173;
                      } else {
                        result[0] += 0.06751161788698398;
                      }
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.002140503787855451;
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.15979053906059715;
                            } else {
                              result[0] += 0.004629902616403367;
                            }
                          }
                        } else {
                          result[0] += 0.02618579283607486;
                        }
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                            result[0] += 0.09097500987995628;
                          } else {
                            result[0] += 0.02451982205302943;
                          }
                        } else {
                          result[0] += -0.12056843442038719;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.0018537596830276557;
                } else {
                  result[0] += -0.054151584755319615;
                }
              } else {
                result[0] += 0.0826420563901498;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.932935476303101474) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.004979233584496297;
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.024207751966068794;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.0339560630972732;
            } else {
              result[0] += -0.022951952303254378;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
          result[0] += 0.00113468010580759;
        } else {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.07123613275506505;
            } else {
              result[0] += -0.015663129060214588;
            }
          } else {
            result[0] += -0.0015676667583012936;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += -0.0015241837904149723;
          } else {
            result[0] += -0.0771207595144809;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.01941280661828286;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.042348374978043134;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.024859972490084942;
              } else {
                result[0] += 0.0027817434697337794;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.305786132812500888) ) ) {
            result[0] += 0.04464715148331966;
          } else {
            result[0] += 0.008132418926690586;
          }
        } else {
          result[0] += -0.0030493111388424834;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.395718574523926669) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.033235127980861354;
          } else {
            result[0] += -0.09039080396513657;
          }
        } else {
          result[0] += 0.008990162145349078;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.0009523123932744583;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.0035693971693616863;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
                result[0] += -0.02153416160350457;
              } else {
                result[0] += -0.048713296331885175;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += -0.02847453783506646;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                result[0] += -0.019132198329913946;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.03120851584539653;
                } else {
                  result[0] += -0.023533386203937398;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.0034173162895515274;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                result[0] += -0.012812880502113844;
              } else {
                result[0] += 0.014437606597784659;
              }
            } else {
              result[0] += -0.0361690360091153;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += 0.007466228951624032;
              } else {
                result[0] += -0.018438992552485357;
              }
            } else {
              if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.73906850814819514) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.059227720750103276;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.02207563256908214;
                          } else {
                            result[0] += -0.009784696260119337;
                          }
                        } else {
                          result[0] += -0.030716265105587122;
                        }
                      } else {
                        result[0] += 0.017251037917574694;
                      }
                    } else {
                      result[0] += 0.021679491962492244;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.007179709338474143;
                  } else {
                    result[0] += 0.07524657739179402;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.11144848919658482;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.07493148016456713;
                    } else {
                      result[0] += -0.011953720840399785;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.46093606948852717) ) ) {
                          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                            result[0] += 0.03433404645519097;
                          } else {
                            result[0] += -0.0029513670896438417;
                          }
                        } else {
                          result[0] += -0.04898493472882106;
                        }
                      } else {
                        result[0] += -0.023216444854796105;
                      }
                    } else {
                      result[0] += 0.015503551868822846;
                    }
                  } else {
                    result[0] += 0.02671028192742172;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.0013028297534690206;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.010901459768400337;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.042641099753394925;
                } else {
                  result[0] += -0.09863794086774125;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                result[0] += -0.04068526437738515;
              } else {
                result[0] += 0.010187851000794666;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.007178416349665467;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                  result[0] += -0.09944773936416715;
                } else {
                  result[0] += -0.01361087352588062;
                }
              } else {
                result[0] += 0.007321079097094895;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
                result[0] += 0.009316901446685273;
              } else {
                result[0] += 0.044986567221950574;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += 0.008607079219598865;
      } else {
        result[0] += -0.02986463823947938;
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
          result[0] += -0.008181799294147096;
        } else {
          result[0] += -0.0249968056063242;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.004011436782972133;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 4.127881952154308e-05;
                } else {
                  result[0] += 0.040113067005815725;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
                    result[0] += 0.022267406252847403;
                  } else {
                    result[0] += -0.01998632692538228;
                  }
                } else {
                  result[0] += -0.01391178445443308;
                }
              }
            } else {
              result[0] += -0.003067247900673548;
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += -0.0451938796090194;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.006427607861834288;
                      } else {
                        result[0] += -0.05246147966418884;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                        result[0] += 0.010425398533398321;
                      } else {
                        result[0] += 0.04374187568252375;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                    result[0] += -0.006705971220317786;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.02337894172885169;
                    } else {
                      result[0] += -0.08849937613750315;
                    }
                  }
                }
              } else {
                result[0] += -0.03996727800094462;
              }
            } else {
              result[0] += 0.0011222467436680001;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.587220668792725498) ) ) {
                  result[0] += 0.0014835863871316085;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06510015734458428;
                  } else {
                    result[0] += 0.019306869189136305;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.0883522033691424) ) ) {
                    result[0] += -0.015673932503989324;
                  } else {
                    result[0] += 0.03637370583268786;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.802100181579590732) ) ) {
                        result[0] += 0.015326596345849647;
                      } else {
                        result[0] += -0.0306541667237869;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += 0.012188678695715096;
                        } else {
                          result[0] += 0.05609213095737973;
                        }
                      } else {
                        result[0] += -0.0284808696105888;
                      }
                    }
                  } else {
                    result[0] += 0.02652144980953664;
                  }
                }
              }
            } else {
              result[0] += -0.03747451294002132;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.003178140416796416;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.233438730239869052) ) ) {
                    result[0] += -0.008490422363178998;
                  } else {
                    result[0] += -0.033156503636807944;
                  }
                } else {
                  result[0] += -0.04194806585999791;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.26464319229126154) ) ) {
                        result[0] += -0.0015005630697739812;
                      } else {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.050247571401766944;
                        } else {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.03288117207078624;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                              result[0] += -0.05615371317521256;
                            } else {
                              result[0] += 0.033567556289169506;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += -0.07969694762574252;
                    }
                  } else {
                    result[0] += 0.018240180997469643;
                  }
                } else {
                  result[0] += 0.04280113231891688;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.385823249816895419) ) ) {
                      result[0] += -0.06101927434542767;
                    } else {
                      result[0] += 0.020280591642255607;
                    }
                  } else {
                    result[0] += 0.005026256540190258;
                  }
                } else {
                  result[0] += 0.006037548423632677;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.005501258647298564;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                result[0] += 0.005045043643089294;
              } else {
                result[0] += 0.03352137873962837;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.01744109585416411;
            } else {
              result[0] += 0.020613017584767256;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.966960191726685458) ) ) {
          result[0] += 0.0026285052797716062;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += 0.02087002900761475;
            } else {
              result[0] += -0.02915144495307779;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += 0.018330165002993143;
            } else {
              result[0] += -0.03575548208702228;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += 0.013849410620127671;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            result[0] += -0.03522863239944964;
          } else {
            result[0] += 0.01377744611460049;
          }
        }
      } else {
        result[0] += -0.0008084750490953698;
      }
    }
  }
  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.913499355316162998) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
          result[0] += 0.004777417817739191;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
            result[0] += -0.0049534658859783295;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.09414350903230623;
            } else {
              result[0] += -0.016445148683347142;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.012913025863175484;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += 0.009150555321137325;
              } else {
                result[0] += -0.038512612057656825;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.03673909317269991;
            } else {
              result[0] += -0.013542756631821954;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.00211572647094904) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.603186130523683417) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.020162893897561165;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.007738299758217403;
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
                          result[0] += 0.009798481620915546;
                        } else {
                          result[0] += 0.06671756646956845;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.015881955356977607;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.015921977597687913;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.847910165786744052) ) ) {
                        result[0] += -0.020603986425589985;
                      } else {
                        result[0] += -0.05932009159091284;
                      }
                    } else {
                      result[0] += 0.0068661468152394645;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                    result[0] += 0.0025238030161584485;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.04581619459628104;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0014650433383697533;
                      } else {
                        result[0] += -0.02810154896285267;
                      }
                    }
                  }
                } else {
                  result[0] += 0.007890571196861181;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.025744639701770617;
              } else {
                result[0] += -0.0004441945341046374;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.03019417715710006;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += 0.06168169254537183;
              } else {
                result[0] += -0.008332661741391465;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
        result[0] += 0.00593295405856645;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
            result[0] += -0.0012462141841372745;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.03223726532887521;
            } else {
              result[0] += -0.04307148941812377;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                result[0] += -0.011860613168952875;
              } else {
                result[0] += -0.0442967167553055;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                result[0] += 0.019914799629053972;
              } else {
                result[0] += -0.03184446312939858;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
              result[0] += 0.00906730288335384;
            } else {
              result[0] += -0.014643754011986016;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.0490408550914897;
        } else {
          result[0] += -0.003312949091740223;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.791781663894654208) ) ) {
          result[0] += -0.04038532447666898;
        } else {
          result[0] += -0.009478570572982063;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          result[0] += 0.06486854364281713;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.012814068461710846;
          } else {
            result[0] += -0.044427635395200164;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.970085620880127397) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.04655212984875837;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0004380472667244913;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.009294553015838763;
                  } else {
                    result[0] += -0.04938460158296671;
                  }
                }
              }
            } else {
              result[0] += 0.0016008033957408043;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                result[0] += 0.0014617282934910275;
              } else {
                result[0] += -0.017203894453189414;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.01249965053144132;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += 0.05652161331823822;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.123651981353760654) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.045262160161011616;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.005508872520887578;
                          } else {
                            result[0] += -0.06389583610767635;
                          }
                        } else {
                          result[0] += 0.020394213172538603;
                        }
                      }
                    } else {
                      result[0] += 0.0019139833054651568;
                    }
                  } else {
                    result[0] += 0.030095831555379777;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.0021150495559871616;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.66339445114135831) ) ) {
              result[0] += -0.05216160353356833;
            } else {
              result[0] += 0.08216543799822396;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.869292974472046787) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.603186130523683417) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.0004812902844563101;
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.028757652220198434;
                  } else {
                    result[0] += -0.0012987619352103539;
                  }
                }
              } else {
                result[0] += 0.01774568764996508;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += 0.016749110855440945;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.03216917996836196;
                } else {
                  result[0] += -0.004274565683865206;
                }
              }
            }
          } else {
            result[0] += 0.028008081256652686;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
            result[0] += 0.04305578482791544;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.04644853883641795;
            } else {
              result[0] += -0.005440970248590173;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
            result[0] += -0.022098984002274155;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.026084693662892552;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.008275341306939785;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.046424161521964755;
                } else {
                  result[0] += 0.009893025901361763;
                }
              }
            }
          }
        } else {
          result[0] += 0.0069130305439933;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
        result[0] += 0.004791905601796733;
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.02378675432375879;
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.067782521247864214) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.06497182551655721;
              } else {
                result[0] += -0.027881121693029295;
              }
            } else {
              result[0] += 0.02726547312258556;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.632926940917970526) ) ) {
                result[0] += 0.0038970982495873797;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += -0.01906584408062352;
                } else {
                  result[0] += 0.019968133390139888;
                }
              }
            } else {
              result[0] += -0.04281866790991914;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.04535745210459533;
        } else {
          result[0] += -0.0037083170517947257;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.966960191726685458) ) ) {
          result[0] += -0.03619909424180658;
        } else {
          result[0] += -0.0036292966479929897;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          result[0] += 0.03802545268488555;
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.013963121152450395;
          } else {
            result[0] += -0.04180196243151055;
          }
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.006409004889742206;
                  } else {
                    result[0] += -0.08124833459941311;
                  }
                } else {
                  result[0] += -0.09924721211384291;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.06440320678456715;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.05882486745246001;
                  } else {
                    result[0] += 0.023436911375740388;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.01026107720768845;
              } else {
                result[0] += 0.007985079784115143;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.07735708490824446;
            } else {
              result[0] += -0.01287104759175354;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                    result[0] += -0.006727221986823508;
                  } else {
                    result[0] += 0.008551848727732724;
                  }
                } else {
                  result[0] += -0.027065281794208362;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += -0.010341513098935459;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.12446265873362204;
                      } else {
                        result[0] += -0.02512824836373097;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.001666845555753741;
                    } else {
                      result[0] += 0.015615223786975328;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.010662439651126692;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += -0.03148796026014585;
                      } else {
                        result[0] += 0.01125473291800696;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79285955429077326) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                          result[0] += -0.009836794677948632;
                        } else {
                          result[0] += 0.023609011693079898;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.03575741032502729;
                        } else {
                          result[0] += -0.0008009772861157692;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.012251988817194337;
              } else {
                result[0] += -0.044080137850236414;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.008651210605194491;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.001864336975118639;
              } else {
                result[0] += 0.03459891122140746;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.338562726974488193) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.008935686843559226;
            } else {
              result[0] += -0.004281106364326312;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.023537426706606714;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.0055874973695915825;
              } else {
                result[0] += -0.025391566034418542;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.019715375206258597;
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.03716887391506495;
            } else {
              result[0] += 0.007717657399339154;
            }
          }
        }
      } else {
        result[0] += -0.009975224938439144;
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0020853317378212;
            } else {
              result[0] += 0.013254941349702374;
            }
          } else {
            result[0] += -0.015209814947184883;
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.02586894587701546;
          } else {
            if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.11336500643272865;
              } else {
                result[0] += 0.010833914486977384;
              }
            } else {
              result[0] += -0.02202245242638817;
            }
          }
        }
      } else {
        result[0] += -0.040291883909603074;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.025703656551657447;
          } else {
            result[0] += 0.03781469373615885;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.623641014099121982) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
              result[0] += 0.0007752621941131364;
            } else {
              result[0] += -0.06975804656227784;
            }
          } else {
            result[0] += -0.0008881411164258887;
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          result[0] += 0.0023076341901688935;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.966960191726685458) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.007651828981080052;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.586156606674195224) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.006867882579342855;
                } else {
                  result[0] += -0.0303121215433893;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  result[0] += -0.057128768244034525;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                    result[0] += -0.05053169091925837;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.011874117966042524;
                    } else {
                      result[0] += -0.058263298618622464;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += -0.011986857882004548;
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.005458209680842145;
              } else {
                result[0] += 0.039854506226146856;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.02148647713244257;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.013215868649686555;
                  } else {
                    result[0] += 0.08040899846052615;
                  }
                }
              } else {
                result[0] += 0.0006669456411562897;
              }
            } else {
              result[0] += 0.00997523622157203;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0038094373494391358;
            } else {
              result[0] += -0.025448727325163192;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.0032518475388445475;
              } else {
                result[0] += 0.04897102970322565;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                    result[0] += -0.035895236667827124;
                  } else {
                    result[0] += 0.031514607405557886;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.02911742901439461;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += -0.0017754058971650695;
                      } else {
                        result[0] += -0.04705174614264312;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += 0.00397964345371449;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.044645806973464805;
                        } else {
                          result[0] += -0.03801867481438234;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.054012682993335594;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.011947789009078419;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.010035604977594348;
                } else {
                  result[0] += -0.022913881526803695;
                }
              } else {
                result[0] += 0.015536705020088859;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.0013777337370791758;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.305786132812500888) ) ) {
                  result[0] += -0.03214209957057714;
                } else {
                  result[0] += -0.004537417892345057;
                }
              } else {
                result[0] += -0.06619756135586351;
              }
            } else {
              result[0] += -0.05898492621118584;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.012748281722205036;
            } else {
              result[0] += -0.01277240046855771;
            }
          } else {
            result[0] += 0.009686881205525244;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
      result[0] += 0.017024042457634114;
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.035837493997919904;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += -0.0008273757201377606;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.004057627135356897;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.07726577322249528;
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.02056349783164481;
                      } else {
                        result[0] += -0.12004872189536145;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.393745899200439897) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.002237332530929098;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                          result[0] += 3.977089991367095e-05;
                        } else {
                          result[0] += 0.06093101325587236;
                        }
                      }
                    } else {
                      result[0] += 0.07932133904009209;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.012604134359485153;
                    } else {
                      result[0] += -0.05261646443290833;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.06444501223451311;
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.1376601848659627;
                    } else {
                      result[0] += -0.02301669963480731;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.0419612621299;
            }
          }
        } else {
          result[0] += -0.03707932499365787;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
            result[0] += 0.0007377742372021087;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.006852078338744284;
            } else {
              result[0] += -0.06318474295134528;
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.013158440936037198;
            } else {
              result[0] += 0.02513107384820308;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.022415161667226915;
                } else {
                  result[0] += -0.052482350011208545;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += -0.009726984749632584;
                  } else {
                    result[0] += 0.024967044316541485;
                  }
                } else {
                  result[0] += -0.0332800491983657;
                }
              }
            } else {
              result[0] += -0.003328955317847464;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.00587335607620074;
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.11731061634926782;
                    } else {
                      result[0] += -0.014830029968160392;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      result[0] += -0.00672290960753541;
                    } else {
                      result[0] += 0.032017422913818316;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += 0.0003588549580892639;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                            result[0] += -0.00500829605760859;
                          } else {
                            result[0] += 0.030191751606884167;
                          }
                        } else {
                          result[0] += -0.00722675551982666;
                        }
                      }
                    } else {
                      result[0] += 0.03743229990963363;
                    }
                  } else {
                    result[0] += -0.01121177851389852;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                  result[0] += 0.0010600332551576105;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.013628412290541694;
                  } else {
                    result[0] += -0.02435140330282315;
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                    result[0] += -0.0008321088816808453;
                  } else {
                    result[0] += -0.02652882048864147;
                  }
                } else {
                  result[0] += -0.08454910719017612;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += 0.010406116747564595;
                  } else {
                    result[0] += -0.006994103215057598;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                    result[0] += 0.007922228089976523;
                  } else {
                    result[0] += 0.033129799307047;
                  }
                }
              } else {
                result[0] += -0.049623627448097514;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.008727295855899347;
              } else {
                result[0] += -0.10379870753223737;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.257122993469240058) ) ) {
            result[0] += -0.011358295097592322;
          } else {
            result[0] += 0.008016061789174127;
          }
        }
      } else {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.082378498318556;
            } else {
              result[0] += 0.23578721407359243;
            }
          } else {
            result[0] += 0.04201785988945096;
          }
        } else {
          result[0] += -0.04725778414604276;
        }
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            result[0] += -0.031185218003611015;
          } else {
            result[0] += 0.009490140137078657;
          }
        } else {
          result[0] += 0.013918349500213431;
        }
      } else {
        result[0] += -0.0006115312712879034;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
            result[0] += -0.0037956867038319454;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += 0.009571233086362922;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.03642678174606189;
              } else {
                result[0] += -0.009300446313906186;
              }
            }
          }
        } else {
          result[0] += -0.000488391628649805;
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.01809282737394935;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                result[0] += 0.04117255884982757;
              } else {
                result[0] += -0.012340967522465813;
              }
            }
          } else {
            result[0] += -0.029056942024108058;
          }
        } else {
          result[0] += 0.003389674784637515;
        }
      }
    } else {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.624251961708069292) ) ) {
            result[0] += -0.01643408766723241;
          } else {
            result[0] += -0.005472412908752852;
          }
        } else {
          result[0] += 0.006631004957987544;
        }
      } else {
        result[0] += -0.0003631112499198701;
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY(  (data[38].missing != -1) && (data[38].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.33441734313965021) ) ) {
            if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
                result[0] += -0.09004698710972621;
              } else {
                result[0] += 0.004904274897137808;
              }
            } else {
              result[0] += -0.0061761429589510235;
            }
          } else {
            result[0] += 0.03637435843475513;
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0006099376484687629;
            } else {
              result[0] += 0.005066003123798624;
            }
          } else {
            result[0] += 0.006830834076599382;
          }
        }
      } else {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.15036104083792365;
          } else {
            result[0] += 0.03525729081740625;
          }
        } else {
          result[0] += -0.043664118243912496;
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.032774426784321964;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += 0.02413937731144519;
            } else {
              result[0] += -0.002386662022395068;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0375142115412878;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.00801609211958979;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.017688733377238752;
                } else {
                  result[0] += 0.05656230637448853;
                }
              }
            } else {
              result[0] += -0.03090508639642093;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.06387792495554012;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.014747466189011042;
                    } else {
                      result[0] += 0.05035550613769385;
                    }
                  }
                } else {
                  result[0] += 0.014426378904563778;
                }
              } else {
                result[0] += -0.004626515434075151;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.9054608345031756) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.004230048800005323;
                  } else {
                    result[0] += 0.044817497964704904;
                  }
                } else {
                  result[0] += -0.030565734387065138;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.046214986409314526;
                } else {
                  result[0] += -0.0011034246276369576;
                }
              }
            }
          } else {
            result[0] += -0.08756189690683937;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                result[0] += -0.05085643878337225;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.305786132812500888) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70956039428711115) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.01833404334944704;
                      } else {
                        result[0] += -0.05584122130004319;
                      }
                    } else {
                      result[0] += 0.0622096849171431;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                      result[0] += -0.07285776567433043;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.13839721679687678) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.069308466245169;
                        } else {
                          result[0] += -0.005914829224853203;
                        }
                      } else {
                        result[0] += 0.04014295035705285;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                      result[0] += -0.023794376397065272;
                    } else {
                      result[0] += 0.031809766347864483;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.025905705489404487;
                      } else {
                        result[0] += 0.08336961513257655;
                      }
                    } else {
                      result[0] += -0.014127200795749027;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                  result[0] += 0.025941321013916153;
                } else {
                  result[0] += -0.01844036859506157;
                }
              } else {
                result[0] += -0.04070158898540794;
              }
            }
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.001563474432845287;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.338562726974488193) ) ) {
                  result[0] += -0.014706303463541896;
                } else {
                  result[0] += 0.027201404661418235;
                }
              } else {
                result[0] += 0.012163655989490386;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.01412578171268506;
      } else {
        result[0] += 0.14319758080754563;
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.031656571649228166;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.645421981811524326) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.007803585545290913;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.297559976577759233) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += 0.00018648250535769934;
                      } else {
                        result[0] += -0.07534768032978185;
                      }
                    } else {
                      result[0] += 0.007595125112173128;
                    }
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.015505020462639095;
                    } else {
                      result[0] += -0.052061481204669094;
                    }
                  }
                }
              } else {
                result[0] += 0.03408377117353819;
              }
            } else {
              result[0] += -0.02039874399166089;
            }
          }
        } else {
          result[0] += -0.034247679535148005;
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
            result[0] += 0.0002236269447438019;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.012377203121693927;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.00873533995369964;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.048825226533171506;
                  } else {
                    result[0] += 0.04331378308189978;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.0396872942963272;
                    } else {
                      result[0] += 0.03215711031108554;
                    }
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.08397640966877637;
                    } else {
                      result[0] += -0.03032913934224706;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.01863623905698991;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.947025299072267401) ) ) {
                        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.050225938123414055;
                        } else {
                          result[0] += 0.021821227308082367;
                        }
                      } else {
                        result[0] += 0.019251849043849564;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.035679812294159015;
                      } else {
                        result[0] += -0.04007642424893228;
                      }
                    } else {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.13952244526761776;
                      } else {
                        result[0] += -0.014507069444944601;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                      result[0] += 0.005612620196035129;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.0261996311659295;
                      } else {
                        result[0] += 0.0028229892460778193;
                      }
                    }
                  } else {
                    result[0] += -0.02679454982689429;
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.0024315318485904672;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.07600602533140924;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.011183347276007841;
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                            result[0] += -7.190976260280255e-05;
                          } else {
                            result[0] += 0.05055421944501947;
                          }
                        } else {
                          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.01597930911998324;
                          } else {
                            result[0] += -0.11865796372999178;
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
          result[0] += -0.015864861535563223;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                    result[0] += -0.012488718402903179;
                  } else {
                    result[0] += 0.01679059287058422;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05479049682617365) ) ) {
                    result[0] += -0.00449696222105083;
                  } else {
                    result[0] += 0.03546357132353863;
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.005719792247347865;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.169590950012207919) ) ) {
                    result[0] += 0.008620200617189613;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += -0.029038235036353013;
                      } else {
                        result[0] += 0.009051394155473508;
                      }
                    } else {
                      result[0] += 0.009373755557349654;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.010950377288741488;
              } else {
                result[0] += -0.003240223810671322;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += -0.013364307888122921;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.014788627624512607) ) ) {
                result[0] += 0.003122862526021536;
              } else {
                result[0] += 0.015470414774876377;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.015333651666923714;
              } else {
                result[0] += -0.08921027525867503;
              }
            } else {
              result[0] += -0.10952521503489963;
            }
          } else {
            result[0] += 0.006950915520248999;
          }
        }
      } else {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.12970495711617158;
          } else {
            result[0] += 0.03048568801482883;
          }
        } else {
          result[0] += -0.04023792397575307;
        }
      }
    } else {
      result[0] += -0.0007821489741406368;
    }
  }
}

