
#include "header.h"

void predict_unit2(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.007433861911176405;
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.651049375534058505) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                    result[0] += -0.011912885359176088;
                  } else {
                    if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.08068183993160045;
                    } else {
                      result[0] += 0.03799036074935633;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.037834912360069195;
                  } else {
                    result[0] += -0.0019110594258570383;
                  }
                }
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.0008089734803854228;
                  } else {
                    result[0] += -0.0794681271302069;
                  }
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.807895898818970615) ) ) {
                      result[0] += 0.0038359388080804907;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.16156268847025274;
                      } else {
                        result[0] += 0.03815940368300007;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
                      result[0] += -0.028792782883325527;
                    } else {
                      result[0] += -0.08178939926501017;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.05517883072754829;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                      result[0] += -0.057681802167242946;
                    } else {
                      result[0] += 0.005599665864320453;
                    }
                  } else {
                    result[0] += 0.014711616394837052;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.597218394279480425) ) ) {
                    result[0] += -0.00360487515551051;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                      result[0] += 0.05056863749744302;
                    } else {
                      result[0] += -0.07217072306821415;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                      result[0] += 0.009163816741321251;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                        result[0] += -0.019063644661249393;
                      } else {
                        result[0] += -0.10286566687618631;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += -0.032885403125627544;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.801661729812622958) ) ) {
                            result[0] += 0.028300543572656368;
                          } else {
                            result[0] += -0.07142392184247273;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
                            result[0] += 0.041713756013253854;
                          } else {
                            result[0] += -0.054155545366585116;
                          }
                        }
                      } else {
                        result[0] += 0.03733159127249252;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.010784352697891528;
                  } else {
                    result[0] += -0.07178562387751872;
                  }
                } else {
                  result[0] += 0.07814223010705734;
                }
              } else {
                result[0] += -0.054848675780881186;
              }
            } else {
              result[0] += 0.006052934629742901;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                result[0] += -0.08979320716910826;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                      result[0] += 0.008623073539133044;
                    } else {
                      result[0] += -0.09215872974616068;
                    }
                  } else {
                    result[0] += -0.06989537063452217;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += -0.11394109322470503;
                  } else {
                    result[0] += 0.021182770570777804;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03119021291470315;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.03808144273044481;
                } else {
                  result[0] += -0.0042084940600062085;
                }
              }
            }
          } else {
            result[0] += 0.003158070402624953;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.014816439505437859;
          } else {
            result[0] += -0.04600855424260763;
          }
        } else {
          result[0] += 0.03481776387320784;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += 0.10164135266255761;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
          result[0] += -0.061970539210745246;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += 0.11268824763432878;
          } else {
            result[0] += 0.03498544311908849;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.0662700339343618;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.11549728195970484;
        } else {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += -0.05598790740490622;
            } else {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.045351826552146556;
              } else {
                result[0] += 0.010559280568467845;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.015966870295994536;
              } else {
                result[0] += -0.03247369964897461;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += -0.03923288483337973;
                } else {
                  result[0] += 0.054384170115831104;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                  result[0] += -0.051163714706061486;
                } else {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.09626172950007142;
                  } else {
                    result[0] += -0.031907345942957;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.64763975143432706) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                    result[0] += -0.0072051641514413595;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.04539604927278587;
                    } else {
                      result[0] += 0.011281315702678638;
                    }
                  }
                } else {
                  result[0] += -0.06144357433814962;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                  result[0] += -0.06197841607452298;
                } else {
                  result[0] += 0.02710834788160725;
                }
              }
            } else {
              result[0] += 0.04111110215035013;
            }
          } else {
            result[0] += -0.00645410786486467;
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
              result[0] += 0.06376736002751679;
            } else {
              if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.11259161080350644;
              } else {
                result[0] += 0.013751593056236619;
              }
            }
          } else {
            result[0] += -0.10617090784107411;
          }
        }
      } else {
        if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.012809541986806852;
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.04748393304148847;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
                    result[0] += 0.018351710566463342;
                  } else {
                    result[0] += -0.018295748531156588;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                result[0] += 0.09753018949308163;
              } else {
                result[0] += -0.13418523008629532;
              }
            }
          } else {
            result[0] += 0.04701152090172181;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
                result[0] += 0.011287196286627678;
              } else {
                result[0] += -0.04617419252730964;
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                result[0] += -0.07636876051501804;
              } else {
                result[0] += 0.054139584510563554;
              }
            }
          } else {
            result[0] += -0.08562541375144883;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
              result[0] += -0.09263394693493004;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.07468996202267274;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)23.50000000000000355) ) ) {
                    if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06272094256668682;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.042225786503858144;
                      } else {
                        result[0] += -0.07115148416542073;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                      result[0] += 0.39534771412998754;
                    } else {
                      result[0] += 0.023281148139641026;
                    }
                  }
                }
              } else {
                result[0] += -0.07266571373608632;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.04512826617962612;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.09303552884774866;
                } else {
                  result[0] += -0.014383130282633362;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.05965175184558839;
              } else {
                result[0] += -0.07305633653703801;
              }
            }
          }
        } else {
          result[0] += -0.12415583024183344;
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += 0.03743996746415285;
                } else {
                  result[0] += -0.03582195435664671;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.006535167123793752;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.07321422685013916;
                  } else {
                    result[0] += -0.018895605805857564;
                  }
                }
              }
            } else {
              result[0] += 0.0014336532691201502;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    result[0] += -0.018320520415579425;
                  } else {
                    result[0] += 0.10285964667032682;
                  }
                } else {
                  result[0] += -0.06527529700886973;
                }
              } else {
                result[0] += 8.555749137329415e-05;
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.05667569623529295;
              } else {
                result[0] += -0.015075011731880989;
              }
            }
          }
        } else {
          result[0] += 0.0009681361491193447;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
      result[0] += -0.0928707471107571;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
            result[0] += -0.0821500420603991;
          } else {
            result[0] += -0.0081381892158055;
          }
        } else {
          result[0] += 0.03244757014465432;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
            result[0] += 0.04930759725408859;
          } else {
            result[0] += -0.0703386158304816;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                result[0] += -0.10952080570168504;
              } else {
                result[0] += -0.006700493917992087;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += -0.051699699010197;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                  result[0] += 0.009162968161127714;
                } else {
                  result[0] += 0.12996700977571282;
                }
              }
            }
          } else {
            result[0] += 0.026977667076300928;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.0032600441355999255;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
          result[0] += -0.005115507330802769;
        } else {
          result[0] += 0.03897896423672995;
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                    result[0] += 0.02139449237232359;
                  } else {
                    result[0] += -0.1445458833276386;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.09174622146154737;
                          } else {
                            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                                result[0] += -0.015593815859477151;
                              } else {
                                result[0] += 0.08550049798002678;
                              }
                            } else {
                              result[0] += -0.10673153222095069;
                            }
                          }
                        } else {
                          result[0] += 0.02166643688739225;
                        }
                      } else {
                        result[0] += -0.09402042716284198;
                      }
                    } else {
                      result[0] += -0.1695649306775204;
                    }
                  } else {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.044493643369935554;
                    } else {
                      result[0] += 0.06733700138542872;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.006812978819367288;
                } else {
                  result[0] += 0.02444951921524204;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.34969997406006037) ) ) {
                result[0] += -0.08487803849031482;
              } else {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.00712672434133636;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.001886371951455719;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += 0.02825749966095197;
                      } else {
                        result[0] += -0.06335483275479978;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.022702960120805818;
                  } else {
                    result[0] += 0.007947333762769883;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.029057501140410908;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.01883937174337786;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                          result[0] += -0.022621976446339755;
                        } else {
                          result[0] += -0.11791503538708632;
                        }
                      } else {
                        result[0] += 0.020483465762123808;
                      }
                    }
                  }
                } else {
                  result[0] += 0.04476464887797714;
                }
              } else {
                result[0] += -0.045739455600639405;
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.015320966707133527;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.06360065995119121;
                } else {
                  result[0] += -0.008231078801721697;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
              result[0] += -0.015844393384559386;
            } else {
              result[0] += -0.08897214400989958;
            }
          } else {
            result[0] += 0.001409536058886318;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.08087366346936675;
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
                result[0] += -0.07436306030553125;
              } else {
                result[0] += 0.05589914610736613;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += 0.03507327123524869;
              } else {
                result[0] += -0.06040394300630031;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.01527766594462709;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.038703924476528456;
                } else {
                  result[0] += -0.022374687877642706;
                }
              } else {
                result[0] += 0.05846404452457321;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += 0.0947573006016835;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
          result[0] += -0.060561126310289495;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += 0.10249260517624269;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
              result[0] += 0.04305880675092308;
            } else {
              result[0] += -0.03316462583364327;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.06347400799112127;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.09288185537199509;
        } else {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += -0.0766355168957081;
            } else {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0788359212073808;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.008049776976188201;
                  } else {
                    result[0] += 0.04718238949245587;
                  }
                }
              } else {
                result[0] += 0.010520289752066531;
              }
            }
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                result[0] += 0.008775039751705956;
              } else {
                result[0] += -0.03558975304499318;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += -0.053818008859176814;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                  result[0] += -0.02358393094495062;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.022354271411311653;
                  } else {
                    result[0] += 0.19172132768398764;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.01932242971754177;
              } else {
                result[0] += -0.012002939027539131;
              }
            } else {
              result[0] += -0.07839264786129613;
            }
          } else {
            result[0] += 0.008570906010137977;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.25930547714233576) ) ) {
            result[0] += 0.00988038475000513;
          } else {
            result[0] += 0.05496911435167589;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.03218125057239902;
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                            result[0] += -0.0022076255771341542;
                          } else {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                              result[0] += 0.137619163966758;
                            } else {
                              result[0] += 0.051121456160476045;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)40.00000000000000711) ) ) {
                            result[0] += -0.049842713464223136;
                          } else {
                            result[0] += 0.0015964380703715787;
                          }
                        }
                      } else {
                        result[0] += -0.08154543168252036;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                      result[0] += 0.056509217790537275;
                    } else {
                      result[0] += -0.05204707571845546;
                    }
                  }
                } else {
                  result[0] += -0.032857619694485106;
                }
              } else {
                result[0] += 0.003676128597071031;
              }
            } else {
              result[0] += 0.09998659138142058;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.08996301510430442;
                        } else {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                            result[0] += -0.006121727788822524;
                          } else {
                            result[0] += 0.07904207145387226;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.10187661525257757;
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.09828894585583586;
                            } else {
                              result[0] += 0.07529195413536782;
                            }
                          }
                        } else {
                          result[0] += -0.07821556757142277;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.015057202541478005;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += 0.10092305947520497;
                        } else {
                          result[0] += 0.006179909578378246;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += 0.021470503867728565;
                      } else {
                        result[0] += -0.08240087187312281;
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                            result[0] += 0.032419720766219864;
                          } else {
                            result[0] += -0.11167140144336718;
                          }
                        } else {
                          result[0] += 0.039497064083396226;
                        }
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.08642777551749779;
                        } else {
                          result[0] += 0.051541011161442475;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.027029019596928716;
                    } else {
                      result[0] += -0.12546715266544817;
                    }
                  } else {
                    result[0] += 0.0061907767673271144;
                  }
                }
              } else {
                result[0] += -0.04252426130093273;
              }
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.04285237211155904;
                } else {
                  result[0] += 0.003787609196827902;
                }
              } else {
                result[0] += -0.009161356292652315;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                result[0] += 0.16867035083463155;
              } else {
                result[0] += -0.014303758510589806;
              }
            } else {
              result[0] += -0.06920496991977045;
            }
          } else {
            result[0] += 0.0011618393719122007;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.016607640647437524;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
              result[0] += -0.07019941541958982;
            } else {
              result[0] += 0.012924390750647708;
            }
          }
        } else {
          result[0] += 0.03403845383077942;
        }
      } else {
        result[0] += -0.08219526760492098;
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
        result[0] += 0.06501321151626872;
      } else {
        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
          result[0] += 0.041409851523774355;
        } else {
          result[0] += -0.030454655921957258;
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.060690614738278206;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.07826266894741901;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.06580699689392508;
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.024819700539629494;
              } else {
                result[0] += 0.004766240211301447;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                result[0] += -0.04283827238251929;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.00560437769156463;
                  } else {
                    result[0] += 0.1729010568332578;
                  }
                } else {
                  result[0] += -0.03931978006084025;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
                        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                          result[0] += 0.002727897011817691;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += -0.10542233706945446;
                            } else {
                              result[0] += -0.01366128747263498;
                            }
                          } else {
                            result[0] += 0.013854240228220968;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += 0.0035886880942087767;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.032030014346894786;
                          } else {
                            result[0] += 0.08486096930441199;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                          result[0] += -0.01027889238006787;
                        } else {
                          result[0] += -0.07208436014325324;
                        }
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += 0.3264386997235225;
                            } else {
                              result[0] += 0.06600548261011491;
                            }
                          } else {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                                result[0] += -0.02379976073693281;
                              } else {
                                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                                  result[0] += -0.005630918289349738;
                                } else {
                                  result[0] += 0.05676553479217603;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                                result[0] += -0.02272679276406213;
                              } else {
                                result[0] += -0.07694530666619986;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                                result[0] += 0.0138530186480336;
                              } else {
                                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                                  result[0] += -0.06603283503946455;
                                } else {
                                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                                    result[0] += 0.05588772954295978;
                                  } else {
                                    result[0] += -0.10347778160573572;
                                  }
                                }
                              }
                            } else {
                              result[0] += 0.04458100032487292;
                            }
                          } else {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                                result[0] += 0.008078222370810175;
                              } else {
                                result[0] += -0.010060453873630254;
                              }
                            } else {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                                result[0] += -0.061337889495828316;
                              } else {
                                result[0] += 0.014384105278960538;
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += 0.05358900882552671;
                    } else {
                      result[0] += -0.06510595297244719;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                      result[0] += 0.03594987885074454;
                    } else {
                      result[0] += 0.12233414308387586;
                    }
                  } else {
                    result[0] += -0.049047261389689435;
                  }
                }
              } else {
                result[0] += -0.03368108907803367;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                result[0] += -0.0011420038263219702;
              } else {
                result[0] += 0.03340445127530231;
              }
            }
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.025456833296303206;
              } else {
                result[0] += -0.10762414221545774;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.74696540832519709) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.744781017303467685) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                        result[0] += 0.011667580869549762;
                      } else {
                        result[0] += 0.0750261940408129;
                      }
                    } else {
                      result[0] += -0.04671279462432343;
                    }
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += -0.0037332351877538126;
                      } else {
                        result[0] += -0.07778800974953842;
                      }
                    } else {
                      result[0] += 0.005849469637104591;
                    }
                  }
                } else {
                  result[0] += -0.05668040891640387;
                }
              } else {
                result[0] += -0.03183714776888362;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.06360755155120228;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                result[0] += -0.0379753284021562;
              } else {
                result[0] += 0.014825364952443673;
              }
            }
          } else {
            result[0] += 0.034086969674605706;
          }
        }
      } else {
        result[0] += 0.14042046437732753;
      }
    } else {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        result[0] += -0.13178936996139146;
      } else {
        result[0] += -0.012327086896490789;
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.08091719599752314;
      } else {
        result[0] += 0.01065470694054469;
      }
    } else {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += 0.16664336710583041;
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
              result[0] += -0.07824586487765396;
            } else {
              result[0] += 0.08934175799049086;
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.07088059746414059;
          } else {
            result[0] += 0.01519676962487587;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
          result[0] += -0.037632420151696835;
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.3367123983260584;
              } else {
                result[0] += -0.07407404694973707;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.009846879224890227;
              } else {
                result[0] += 0.19234742872763688;
              }
            }
          } else {
            result[0] += -0.03855564018513828;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0034591986176141344;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.13587186052635072;
            } else {
              result[0] += -0.06185071248917206;
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
              result[0] += -0.0068235811206550845;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.12733796661788613;
                } else {
                  result[0] += 0.045195627882160826;
                }
              } else {
                result[0] += 0.021683888831335457;
              }
            }
          } else {
            result[0] += 0.08037521255161223;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
            result[0] += 0.018348756838183208;
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.09436390593015156;
            } else {
              result[0] += 0.022900742922645087;
            }
          }
        } else {
          result[0] += -0.05687162892867523;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += -0.00376565914778736;
          } else {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.08710981487975468;
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += -0.09540757946510563;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.08449332983602192;
                  } else {
                    result[0] += 0.08248794541615598;
                  }
                } else {
                  result[0] += -0.08156434250934597;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.07472965754365794;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.08983289405956235;
                  } else {
                    result[0] += -0.10023804825031371;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
                      result[0] += 0.04073906859148829;
                    } else {
                      result[0] += -0.03801251450254694;
                    }
                  } else {
                    result[0] += 0.07187577395700982;
                  }
                }
              } else {
                result[0] += -0.02419987026575378;
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.0701001616589231;
              } else {
                result[0] += 0.045053914637015066;
              }
            } else {
              result[0] += -0.06653824353951442;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.06450314019911141;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += 0.07140945230942551;
              } else {
                result[0] += -0.0009816627232021328;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.042578442877242564;
              } else {
                result[0] += 0.003855506595088535;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
            result[0] += -0.0018902073437187528;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.285887241363526279) ) ) {
                result[0] += -0.0674257984372049;
              } else {
                result[0] += 0.017840296460721352;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.580392837524414951) ) ) {
                  result[0] += 0.048842553462919665;
                } else {
                  result[0] += -0.03281296254350705;
                }
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.11013688929759968;
                } else {
                  result[0] += 0.018952473349911736;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.021788144616837264;
      } else {
        result[0] += 0.06055067440747119;
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.05644247140454083;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.05992385508309809;
        } else {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.02276447819041641;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.03916844020981692;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                      result[0] += -0.022643651172619024;
                    } else {
                      result[0] += -0.11201927377445627;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += -0.07444344447251512;
                  } else {
                    result[0] += 0.019437953220765806;
                  }
                }
              }
            } else {
              result[0] += -0.02829194404743901;
            }
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.05174843298500026;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += -0.06208589888546141;
                  } else {
                    result[0] += 0.0078042357466547585;
                  }
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += 0.019040155761925574;
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                        result[0] += 0.07265473303072936;
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 1.3302756963456315;
                          } else {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                              result[0] += 0.03394103837727116;
                            } else {
                              result[0] += 0.8820100953481221;
                            }
                          }
                        } else {
                          result[0] += 0.053798423532956885;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.10224714267880644;
                    } else {
                      result[0] += 0.028234954563274878;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.009853002825402399;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              result[0] += -0.07163078547803516;
            } else {
              result[0] += 0.006921893197342505;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.69332504272461115) ) ) {
              result[0] += 0.009298594030115346;
            } else {
              result[0] += 0.0532405246711675;
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.018740498198520233;
          } else {
            result[0] += -0.014744872224824793;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                result[0] += -0.00021704807729812904;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += -0.06979864601708534;
                } else {
                  result[0] += -0.008348214827199406;
                }
              }
            } else {
              result[0] += 0.09197571456771739;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.09290329527707843;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.09721719220562743;
                              } else {
                                result[0] += -0.09785392046193964;
                              }
                            } else {
                              result[0] += 0.06142345137705932;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.08660963263710528;
                          } else {
                            result[0] += -0.00881919833627976;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += -0.04968236782891;
                          } else {
                            result[0] += 0.06609946396390158;
                          }
                        } else {
                          result[0] += -0.07361684019582616;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.014907839914746927;
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                          result[0] += 0.06529124431559251;
                        } else {
                          result[0] += -0.0348778612406986;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += 0.017410806277872597;
                      } else {
                        result[0] += -0.08122413035355781;
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.07508315460040674;
                          } else {
                            result[0] += -0.0007395169801112733;
                          }
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                              result[0] += 0.03883906269707753;
                            } else {
                              result[0] += -0.10198670805332173;
                            }
                          } else {
                            result[0] += 0.008405705658299893;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.0853025388800146;
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.1124538711621572;
                            } else {
                              result[0] += 0.0020046886434897777;
                            }
                          } else {
                            result[0] += 0.08611424939324464;
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.02834070512215869;
                }
              } else {
                result[0] += -0.04253361652261156;
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                        result[0] += -0.09595802014344779;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.06997284190388918;
                        } else {
                          result[0] += 0.004522617992331964;
                        }
                      }
                    } else {
                      result[0] += -0.01663003601606758;
                    }
                  } else {
                    result[0] += -0.006997472962576157;
                  }
                } else {
                  result[0] += 0.017527807301396976;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.06527852372234012;
                } else {
                  result[0] += -0.009776782357467283;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
              result[0] += -0.01214193271642564;
            } else {
              result[0] += -0.06532659131683244;
            }
          } else {
            result[0] += 0.0009713288548135759;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.11329207402957105;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += 0.04872783637462053;
              } else {
                result[0] += -0.007507971576671317;
              }
            }
          } else {
            result[0] += -0.042453184236055264;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.02715028959051079;
          } else {
            result[0] += 0.038701632233341526;
          }
        }
      } else {
        result[0] += -0.07998426966282052;
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
        result[0] += 0.05606778908010939;
      } else {
        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
          result[0] += 0.03304137676777472;
        } else {
          result[0] += -0.0326034438073454;
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.055378716479999104;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
            result[0] += 0.028053292943065147;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
              result[0] += 0.07996984842578228;
            } else {
              result[0] += -0.04461003443568695;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.06667278357496798;
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.07489233357529396;
            } else {
              result[0] += 0.0036579720977621705;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.117121219635010654) ) ) {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += 0.032693619833769276;
                      } else {
                        result[0] += -0.08689859677639329;
                      }
                    } else {
                      result[0] += 0.002350952655354612;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                      result[0] += 0.055464711171489794;
                    } else {
                      result[0] += -0.060094899021056775;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.010604343792430637;
                  } else {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                      result[0] += 0.10565419924421154;
                    } else {
                      result[0] += 0.03014220913569604;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.06292265335052671;
                } else {
                  result[0] += 0.07843637731738927;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.019762324694321678;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += -0.09073050393321563;
                  } else {
                    result[0] += 0.15501597798415767;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                        result[0] += 0.011295145793824233;
                      } else {
                        result[0] += 0.16697134201209027;
                      }
                    } else {
                      result[0] += -0.039150466792233406;
                    }
                  } else {
                    result[0] += -0.05830609892419584;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += 0.03025199915040966;
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.03226332965461854;
                  } else {
                    result[0] += -0.05790352322943801;
                  }
                } else {
                  result[0] += -0.08781153673924909;
                }
              } else {
                result[0] += 0.04825702773076165;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                result[0] += -0.008653908333764655;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += -0.1276572269340313;
                } else {
                  result[0] += 0.012463863392111004;
                }
              }
            } else {
              result[0] += 0.05682553277743797;
            }
          } else {
            result[0] += 0.07365703103273094;
          }
        }
      } else {
        if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                if ( LIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.009769108361868712;
                } else {
                  result[0] += 0.037766777340324;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                  result[0] += -0.1098925442192465;
                } else {
                  result[0] += 0.008162740005356748;
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += -0.07277522845412014;
              } else {
                result[0] += 0.018345327312940283;
              }
            }
          } else {
            result[0] += -0.05856757072206412;
          }
        } else {
          result[0] += 0.08656345094980916;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
          result[0] += 0.049491216379237535;
        } else {
          if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.038826328624437384;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.65906000137329146) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += -0.10241501552710504;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.500490188598633701) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
                      result[0] += 0.0021370471708398206;
                    } else {
                      result[0] += -0.06739943544918608;
                    }
                  } else {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                      result[0] += -0.0013640845677859528;
                    } else {
                      result[0] += 0.2816581693583841;
                    }
                  }
                }
              } else {
                result[0] += 0.1727214178551425;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.04725332691406842;
              } else {
                result[0] += 0.056417368099816484;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.08502656272908082;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.016038384573473917;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                      result[0] += 0.07208870373517025;
                    } else {
                      result[0] += -0.08916079878082217;
                    }
                  }
                } else {
                  result[0] += 0.04373441771328912;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
            result[0] += -0.0020215654158569356;
          } else {
            result[0] += -0.05123315804403581;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.04286080277511306;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.783419609069825995) ) ) {
              result[0] += 0.0300183176751066;
            } else {
              result[0] += -0.0738914326328751;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        result[0] += -0.1152676867634638;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.032589236340177845;
        } else {
          result[0] += 0.035128090645152675;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.039839935058745966;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.07196825906356795;
            } else {
              result[0] += -0.017125071244130523;
            }
          } else {
            result[0] += 0.06508829214091914;
          }
        } else {
          result[0] += 0.021227102227727834;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += 0.03185858525025363;
      } else {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            result[0] += 0.00565991690028689;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
              result[0] += 0.00446651033486171;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                result[0] += 0.024573833706843948;
              } else {
                result[0] += 0.07854820373886658;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06052305230804034;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.07816767480229864;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.04557504084874885;
                  } else {
                    result[0] += 0.026942790349176376;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.03438489070487479;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.012579779673647734;
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.0693230300535587;
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.04946383845324287;
                          } else {
                            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.025118483577822894;
                            } else {
                              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                                result[0] += 0.06771134251068024;
                              } else {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                                  result[0] += -0.04373188337779904;
                                } else {
                                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                                    result[0] += 0.05037644836604656;
                                  } else {
                                    result[0] += -0.02090194898426448;
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.08293387682946887;
                              } else {
                                result[0] += 0.0044086215052679635;
                              }
                            } else {
                              result[0] += 0.03356925311417169;
                            }
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                              result[0] += 0.02268790670309037;
                            } else {
                              result[0] += -0.06545201604171484;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.09618495349825724;
                          } else {
                            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.11318270535561772;
                            } else {
                              result[0] += -0.008971657715667377;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                          result[0] += -0.11893445855002965;
                        } else {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += 0.00661475913506974;
                            } else {
                              result[0] += -0.07806551535177313;
                            }
                          } else {
                            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                              result[0] += 0.03401623917573978;
                            } else {
                              result[0] += -0.04369621349557368;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.04852264763495802;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.016935820797619735;
                      } else {
                        result[0] += 0.07846119121465979;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.032234065257109486;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
          result[0] += 0.03998802208877683;
        } else {
          result[0] += -0.03503381525038992;
        }
      } else {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
          result[0] += 0.06649121543012586;
        } else {
          result[0] += 0.02315127069636974;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.07020142091936482;
            } else {
              if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                result[0] += 0.05108421769100477;
              } else {
                result[0] += 0.333467746209287;
              }
            }
          } else {
            result[0] += -0.07543765561211974;
          }
        } else {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += -0.06831748328366483;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += 0.09111286488677005;
            } else {
              result[0] += -0.08037259302224958;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0598637195436561;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                result[0] += 0.012353002710849227;
              } else {
                result[0] += 0.15675403675695243;
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.04676576831094398;
                } else {
                  result[0] += -0.055229560535936174;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.07018279465913925;
                } else {
                  result[0] += -0.06886343813342374;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                result[0] += 0.13293042267908192;
              } else {
                result[0] += 0.021140481170847568;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)27.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += 0.07472670860325546;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                  result[0] += -0.07105491504096696;
                } else {
                  result[0] += 0.026064757473487722;
                }
              } else {
                result[0] += -0.09702932315735702;
              }
            }
          } else {
            result[0] += 0.1642327879551784;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0005275297641083135;
        } else {
          result[0] += -0.0816562378766796;
        }
      } else {
        result[0] += -0.0008353551554503785;
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.004064190511728208;
    } else {
      result[0] += 0.02284192951800269;
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
              result[0] += -0.042069109405141705;
            } else {
              result[0] += 0.06629408862164586;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.1544354195627521;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
                  result[0] += 0.08797051324855809;
                } else {
                  result[0] += -0.004563779249970436;
                }
              }
            } else {
              result[0] += 0.15279251928297416;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                  result[0] += -0.06412219775620193;
                } else {
                  result[0] += -0.01713369631998534;
                }
              } else {
                result[0] += 0.17714610666083938;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += -0.06390208369977889;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.046675851125767234;
                } else {
                  result[0] += 0.08603886127357234;
                }
              }
            }
          } else {
            result[0] += -0.0697426578714408;
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.024615406216983254;
            } else {
              result[0] += 0.019314732838266548;
            }
          } else {
            result[0] += -0.01165668465478567;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            result[0] += 0.02004386492448735;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.04420709475241412;
                  } else {
                    result[0] += -0.09220639826900834;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                    result[0] += -0.006939599577907918;
                  } else {
                    result[0] += -0.06588262848481774;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += -0.04150353847815371;
                } else {
                  result[0] += -0.0024453844252350482;
                }
              }
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += -0.11451707051176928;
                } else {
                  result[0] += 0.014352758708607706;
                }
              } else {
                result[0] += 0.05627116246307942;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0563655387611213;
            } else {
              result[0] += -0.11888336534774119;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
              result[0] += 0.009363427616726703;
            } else {
              result[0] += 0.06451939604828029;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.12261826433822705;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.011785470370170394;
                } else {
                  result[0] += -0.051969897252791365;
                }
              }
            } else {
              result[0] += -0.05542723321869513;
            }
          } else {
            result[0] += -0.08628853184120144;
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += -0.10055634146706581;
        } else {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.012679749392274179;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += -0.005325538996843828;
                            } else {
                              result[0] += 0.06200268452893897;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                            result[0] += -0.0149070330859409;
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += 0.007961349267014781;
                            } else {
                              result[0] += 0.05599017938212373;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.047676195365887256;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.0945843877636682;
                        } else {
                          result[0] += -0.06248811651724881;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                          result[0] += 0.02737735849639609;
                        } else {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.006441477384723006;
                          } else {
                            result[0] += 0.042511078628184845;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.11191282984603688;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.07204703374916364;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                            result[0] += 0.0002643012277924424;
                          } else {
                            result[0] += 0.03718291266669711;
                          }
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                              result[0] += -0.10793102179034715;
                            } else {
                              result[0] += -0.005365206468964989;
                            }
                          } else {
                            result[0] += -0.09803655659576288;
                          }
                        }
                      } else {
                        result[0] += -0.032492077392942215;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.10020855610627762;
                    } else {
                      result[0] += 0.01372870784593101;
                    }
                  }
                }
              } else {
                result[0] += -0.05240062067673517;
              }
            } else {
              result[0] += 0.021966590548690726;
            }
          } else {
            result[0] += 0.01920102634535505;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.084203958511353427) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                result[0] += 0.006395528484373275;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.02773384409719272;
                  } else {
                    result[0] += -0.104829845565191;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.05584690935912208;
                  } else {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.014571548793221856;
                    } else {
                      result[0] += 0.08815298031622953;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.11590668747579524;
              } else {
                result[0] += 0.05732831621494297;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.07265984483263326;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.10441451581118022;
                    } else {
                      result[0] += 0.0026692676009917246;
                    }
                  } else {
                    result[0] += 0.08360924975596763;
                  }
                } else {
                  result[0] += -0.037600536359035486;
                }
              }
            } else {
              result[0] += -0.059412917311203496;
            }
          }
        } else {
          if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.016018561242651688;
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += -0.030165334064606975;
              } else {
                result[0] += -0.0820731890810178;
              }
            } else {
              result[0] += 0.05106616259660225;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.01750056714141711;
            } else {
              result[0] += 0.061644575952229455;
            }
          } else {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03224027341597186;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0706922956224747;
                  } else {
                    result[0] += -0.04317170057024041;
                  }
                }
              } else {
                result[0] += -0.05806214593367538;
              }
            } else {
              result[0] += 0.08517970065749088;
            }
          }
        } else {
          result[0] += 0.09209034074368748;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.0989725680763609;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.11325960171632798;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
            if ( UNLIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0677642226328295;
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                result[0] += 0.03027727858737124;
              } else {
                result[0] += -0.10287823547129758;
              }
            }
          } else {
            result[0] += -0.0813722228224293;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.552201986312867099) ) ) {
            result[0] += -0.06753626295863972;
          } else {
            result[0] += 0.09103340720130498;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += -0.001810170073895134;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.03992300674678381;
                  } else {
                    result[0] += 0.17525914535390752;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                  result[0] += 0.055881926643336105;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += -0.09585770594446044;
                  } else {
                    result[0] += 0.13384431068618702;
                  }
                }
              }
            } else {
              result[0] += -0.060957262373527235;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
              result[0] += -0.05820573392143267;
            } else {
              result[0] += 0.0020589329981556934;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          result[0] += 0.015260577889762789;
        } else {
          result[0] += 0.12582979658475363;
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.65882921218872248) ) ) {
              result[0] += 0.001994375286161733;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += -0.0341517656208013;
                } else {
                  result[0] += -0.0011529374129719088;
                }
              } else {
                result[0] += 0.005983351855773671;
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                result[0] += -0.059632726370979675;
              } else {
                result[0] += 0.026472279331668926;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.032094853177752396;
              } else {
                result[0] += 0.02967645215263033;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.09063840917101636;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.04548233460614173;
                  } else {
                    result[0] += -0.13597948602213614;
                  }
                } else {
                  result[0] += 0.021618574808339324;
                }
              } else {
                result[0] += 0.02713288765125779;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                result[0] += 0.03676286726065444;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.1677098499889411;
                } else {
                  result[0] += -0.08769886548388578;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.08492710083138838;
        } else {
          result[0] += -0.014782035348041154;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.005056407216146684;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
          result[0] += 0.004282322913430684;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.02406207601175271;
          } else {
            result[0] += 0.07559055103750173;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += -0.002227521054970548;
                } else {
                  result[0] += 0.08199777945776994;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += -0.02064190009293164;
                      } else {
                        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                            result[0] += 0.005370172717494053;
                          } else {
                            result[0] += 0.07658864828503835;
                          }
                        } else {
                          result[0] += -0.06118335219262745;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += 0.03600631430697567;
                          } else {
                            result[0] += -0.0276243169424292;
                          }
                        } else {
                          result[0] += 0.04207637109262547;
                        }
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                            result[0] += 0.001401012785628454;
                          } else {
                            result[0] += -0.08192565917873322;
                          }
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.058814568497821684;
                          } else {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += -0.08357972192087897;
                              } else {
                                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                                  result[0] += -0.06467638997141742;
                                } else {
                                  result[0] += 0.10318374026976376;
                                }
                              }
                            } else {
                              result[0] += 0.0636238965165906;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.04390322478830962;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.02440935894552917;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.06570231562210889;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.05882465291411096;
                        } else {
                          result[0] += 0.09511676179801536;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.006387303519128605;
                    } else {
                      result[0] += 0.06814499078170143;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += -0.013684072741930002;
                } else {
                  result[0] += -0.06971434050085261;
                }
              } else {
                result[0] += 0.0015976314865624765;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.04406491451709824;
                    } else {
                      result[0] += -0.1574943943124869;
                    }
                  } else {
                    result[0] += 0.06230492489506693;
                  }
                } else {
                  result[0] += -0.04624516806848322;
                }
              } else {
                result[0] += 0.01062560480601111;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += 0.020320338951230723;
                } else {
                  result[0] += -0.0795106974255701;
                }
              } else {
                result[0] += -0.05467385146551699;
              }
            }
          }
        } else {
          result[0] += -0.046134297236187956;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.07938715754944276;
        } else {
          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.05653563548198799;
            } else {
              result[0] += 0.10215676657077838;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
              result[0] += -0.04228827049538782;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                result[0] += -0.08419160118039647;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.06391456454998379;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                      result[0] += -0.1058789506179976;
                    } else {
                      result[0] += 0.03170069898888948;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                      result[0] += 0.03471230281056895;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.08530482126378791;
                      } else {
                        result[0] += -0.001708202363422571;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += 0.06810264536483276;
                      } else {
                        result[0] += -0.06652259451985175;
                      }
                    } else {
                      result[0] += 0.035544918620185474;
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
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        result[0] += -0.06861320464956236;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.02604279729825404;
        } else {
          result[0] += 0.04082136679428054;
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.048643198697324176;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.06830783132186972;
          } else {
            result[0] += 0.019666960589812467;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
                result[0] += -0.06577292424215554;
              } else {
                result[0] += 0.009263523399174228;
              }
            } else {
              result[0] += 0.08178382070733202;
            }
          } else {
            result[0] += 0.026042483862444246;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.69332504272461115) ) ) {
            result[0] += 0.0006456960256934535;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
                result[0] += 0.014231770544045659;
              } else {
                result[0] += -0.010842038271893551;
              }
            } else {
              result[0] += -0.09786134233150094;
            }
          }
        } else {
          result[0] += -0.11245641580404123;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.02124070830536008;
            } else {
              result[0] += -0.05195037819145592;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += 0.018075064357865912;
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)13.50000000000000178) ) ) {
                  result[0] += -0.02215547098952521;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                    result[0] += -0.1422843043115559;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                      result[0] += 0.054192362521479236;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
                        result[0] += -0.0627629464442419;
                      } else {
                        result[0] += -0.3133017004323133;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.0378839851990281;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
            result[0] += 7.00801909618613e-05;
          } else {
            result[0] += 0.11205565899781136;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.90474271774292081) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.013888495471643473;
            } else {
              result[0] += -0.08376959388232132;
            }
          } else {
            result[0] += 0.025660344854388707;
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.014058991508880528;
            } else {
              result[0] += -0.04811810023478585;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
              result[0] += -0.05376968471800752;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += -0.03581643301653643;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.038515133578982785;
                } else {
                  result[0] += 0.14288762382986317;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                  result[0] += -0.03403504226682723;
                } else {
                  result[0] += 0.07150394947949375;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += 0.0981195645062135;
                } else {
                  result[0] += -0.004716636409001114;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.0030473307435623094;
              } else {
                result[0] += 0.16106196982471507;
              }
            }
          } else {
            result[0] += -0.003254312952761388;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
            result[0] += -0.05074684130349802;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.006902889914713682;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.053058463080534496;
                } else {
                  result[0] += 0.08840215814810973;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.475347042083742011) ) ) {
                    result[0] += 0.033769927545413844;
                  } else {
                    result[0] += -0.048261648275936396;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.10079433546220776;
                    } else {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.04198431668730827;
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                            result[0] += -0.09284829314778238;
                          } else {
                            result[0] += 0.0688258783612509;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += -0.02993342773397148;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                                  result[0] += 0.10910113400740402;
                                } else {
                                  result[0] += 0.011632309008701502;
                                }
                              } else {
                                result[0] += -0.08596568565870741;
                              }
                            } else {
                              result[0] += -0.05548597420567223;
                            }
                          } else {
                            result[0] += -0.015365350952627133;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.007076131108084531;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += -0.11826235010034157;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                    result[0] += -0.03422025227309927;
                  } else {
                    result[0] += 0.08756116119479443;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
      result[0] += -0.03132656929502717;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += -0.03973893111206819;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
            result[0] += 0.05105950034761441;
          } else {
            result[0] += -0.0431797378695217;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                    result[0] += -0.13064234976167763;
                  } else {
                    result[0] += -0.04464824105950774;
                  }
                } else {
                  result[0] += -0.0074477835148716994;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.06478504425396696;
                } else {
                  result[0] += 0.023369907573481296;
                }
              }
            } else {
              result[0] += 0.08216720907511604;
            }
          } else {
            result[0] += 0.02951947174201269;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.015796868041576165;
                  } else {
                    result[0] += -0.013442888371690035;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.10743980053411678;
                  } else {
                    result[0] += 0.02242336213911577;
                  }
                }
              } else {
                result[0] += 0.007213822203530672;
              }
            } else {
              result[0] += 0.022252168509245957;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.0006941946332168101;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.025345471426070423;
                      } else {
                        result[0] += -0.08888429210152508;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                            result[0] += 0.09387146677241771;
                          } else {
                            result[0] += 0.018090446042721395;
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.6749763488769549) ) ) {
                            result[0] += 0.014302946442295822;
                          } else {
                            result[0] += -0.05208560537062251;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.03321202576580584;
                          } else {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                  result[0] += 0.026317489438590448;
                                } else {
                                  result[0] += -0.03613066833402723;
                                }
                              } else {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                                  result[0] += 0.037502937031731874;
                                } else {
                                  result[0] += -0.051347191266298164;
                                }
                              }
                            } else {
                              result[0] += 0.017464131902985926;
                            }
                          }
                        } else {
                          result[0] += -0.04921792868363565;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.07957954931117985;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += -0.007641474321080877;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.07450530886215462;
                  } else {
                    result[0] += -0.024331505477213865;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.09114294626572106;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += -0.0623099173181769;
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.012917966745867948;
                      } else {
                        result[0] += 0.04528846735206407;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.05194358493535726;
                        } else {
                          result[0] += 0.00517101680486267;
                        }
                      } else {
                        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.044097403820142635;
                        } else {
                          result[0] += 0.00027651781506260914;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.00237754993542447;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                    result[0] += 0.078718958530766;
                  } else {
                    result[0] += -0.08116743391572882;
                  }
                } else {
                  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.09008363179420492;
                  } else {
                    result[0] += 0.03547889875377602;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.03400753058605964;
                } else {
                  result[0] += 0.005824906318517364;
                }
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.03236560516265465;
                } else {
                  result[0] += -0.04277216047848479;
                }
              } else {
                result[0] += -0.08700582598432306;
              }
            }
          } else {
            result[0] += -0.041945217756608956;
          }
        }
      } else {
        result[0] += -0.04376939634039667;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.08032906670642287;
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.08104074364581706;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.2372497548291769;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += -0.053352205241438834;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                result[0] += -0.04997397674435902;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.05717912550943036;
                  } else {
                    result[0] += -0.008390977257309083;
                  }
                } else {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += 0.02854433448715791;
                    } else {
                      result[0] += -0.09184973330248629;
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += 0.0535528696167522;
                      } else {
                        result[0] += -0.06439440702182804;
                      }
                    } else {
                      result[0] += 0.02952084117136359;
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
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
      result[0] += -0.07857555095379512;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.038795087347866096;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
              result[0] += -1.0164651384918297e-05;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.08724415964000198;
              } else {
                result[0] += 0.037793569297908274;
              }
            }
          } else {
            result[0] += 0.014140284912480889;
          }
        } else {
          result[0] += -0.00500326918048047;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.006451453727008317;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.736160039901734287) ) ) {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.008193533947406679;
                      } else {
                        result[0] += 0.09272188853296406;
                      }
                    } else {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.08242695229687624;
                      } else {
                        result[0] += -0.019890451458997194;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.08056568771151904;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.05226887118036991;
                      } else {
                        result[0] += -0.051069635131446446;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.007001799874631386;
                  } else {
                    result[0] += 0.04425601802697931;
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                                result[0] += 0.0026886843816038993;
                              } else {
                                result[0] += -0.05833574998091991;
                              }
                            } else {
                              result[0] += 0.06850542648432316;
                            }
                          } else {
                            result[0] += -0.07033763984182174;
                          }
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
                            result[0] += 0.11741604574001777;
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.22539683613161043;
                            } else {
                              result[0] += -0.08994618859770694;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += 0.002275321121371034;
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                                  result[0] += 0.02998753454273297;
                                } else {
                                  result[0] += -0.0442147119838435;
                                }
                              } else {
                                result[0] += -0.04120314072858227;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                              result[0] += -0.1279938071465702;
                            } else {
                              result[0] += 0.116192437775898;
                            }
                          }
                        } else {
                          result[0] += -0.03920836742343822;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                          result[0] += 0.021881358599248106;
                        } else {
                          result[0] += -0.08665027956009005;
                        }
                      } else {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                            result[0] += 0.021460628618974895;
                          } else {
                            result[0] += -0.05968474382010911;
                          }
                        } else {
                          result[0] += 0.06339438094426003;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.024239160169373174;
                  }
                }
              }
            } else {
              result[0] += -0.007519498965240365;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
              result[0] += 0.034401787794366104;
            } else {
              result[0] += -0.04020504135592212;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06824463294947113;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.08330653630010192;
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.02816849076565608;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.09182960270219541;
                    } else {
                      result[0] += 0.12470717821938422;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.07219233037002025;
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.076962471008301669) ) ) {
                        result[0] += 0.013419439742348866;
                      } else {
                        result[0] += 0.11868858651965666;
                      }
                    } else {
                      result[0] += -0.060369852271809715;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.02624773446880911;
                    } else {
                      result[0] += 0.12430670284154002;
                    }
                  } else {
                    result[0] += -0.07690311683653837;
                  }
                }
              }
            }
          } else {
            result[0] += -0.002532675545798375;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
          result[0] += -0.005517562365138406;
        } else {
          result[0] += -0.06570942506086265;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.07743104486219979;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.03169616291869288;
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.09976666260344659;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                result[0] += 0.04911630782936608;
              } else {
                result[0] += -0.05643306629097333;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += 0.03053031991661083;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
                    result[0] += -0.060271845919569245;
                  } else {
                    result[0] += 0.050117508653474564;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.057150124260395035;
                  } else {
                    result[0] += -0.051827891698983934;
                  }
                } else {
                  result[0] += 0.02918228455050352;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
      result[0] += -0.0290165566884602;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        result[0] += -0.09626372954044665;
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.024073990058919133;
        } else {
          result[0] += -0.0022418693199774005;
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.433569431304932529) ) ) {
            result[0] += 0.006358055272153858;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.013256793997046752;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                    result[0] += -0.006332874153157639;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.023472158038538245;
                    } else {
                      result[0] += -0.08030655552080075;
                    }
                  }
                } else {
                  result[0] += 0.01024069658729583;
                }
              }
            } else {
              result[0] += -0.05179794392527031;
            }
          }
        } else {
          if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.2687106132507342) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.007123766910694745;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    result[0] += -0.013814181599309253;
                  } else {
                    result[0] += -0.1067914244178929;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.02798641357924484;
                  } else {
                    result[0] += -0.13326762235228384;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.038971270379388245;
              } else {
                result[0] += -0.03103767301194159;
              }
            }
          } else {
            result[0] += 0.05858089595786406;
          }
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.042435407638550693) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                    result[0] += 0.006584686300819039;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.05678810259448624;
                    } else {
                      result[0] += 0.018788461872045293;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.04841157430009202;
                  } else {
                    result[0] += -0.11527392360204386;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                  result[0] += 0.011852690793987129;
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.208071470260621005) ) ) {
                        result[0] += -0.05053831822636691;
                      } else {
                        result[0] += -0.011967140834138445;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += -0.0465370648731583;
                        } else {
                          result[0] += 0.03943683564544068;
                        }
                      } else {
                        result[0] += -0.019857516189574154;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      result[0] += 0.010394558106949935;
                    } else {
                      result[0] += -0.028419400157761565;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.64763975143432706) ) ) {
                      result[0] += 0.0012545222925379655;
                    } else {
                      result[0] += -0.08795524009136402;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
                        result[0] += -0.04958715684158387;
                      } else {
                        result[0] += 0.03678207276210002;
                      }
                    } else {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += -0.062411544221212534;
                        } else {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                            result[0] += 0.0062005768506083435;
                          } else {
                            result[0] += 0.11192280297520658;
                          }
                        }
                      } else {
                        result[0] += -0.058084575883455114;
                      }
                    }
                  }
                } else {
                  result[0] += -0.014351795602099319;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += -0.07146976703041795;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += -0.010708224692281824;
                    } else {
                      result[0] += -0.05256308512129154;
                    }
                  } else {
                    result[0] += 0.02576594548081474;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.2687106132507342) ) ) {
                result[0] += 0.014702399176729384;
              } else {
                result[0] += -0.05754069516436444;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.38936424255371271) ) ) {
                  result[0] += -0.07622937859393256;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.65486812591552912) ) ) {
                      result[0] += -0.1263418697149264;
                    } else {
                      result[0] += 0.09764731046117364;
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += 0.045399399891641275;
                    } else {
                      result[0] += -0.06760464603352966;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                    result[0] += 0.025501475816911586;
                  } else {
                    result[0] += -0.13811576445364632;
                  }
                } else {
                  result[0] += -0.11887374595104566;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.09189958563790207;
          } else {
            result[0] += -0.017014431788351998;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.03252647642237055;
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.08944549689384196;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            result[0] += -0.11397209598973948;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                  result[0] += 0.055002904630749076;
                } else {
                  result[0] += -0.0008390907904068143;
                }
              } else {
                result[0] += -0.04740254609114852;
              }
            } else {
              result[0] += 0.03168909339382394;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.02863434488476693;
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.015552951051112252;
      } else {
        result[0] += 0.01519333920582699;
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.01565264897066891;
                  } else {
                    result[0] += -0.012238702679182416;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.14404866505615685;
                  } else {
                    result[0] += -0.04080956188312798;
                  }
                }
              } else {
                result[0] += 0.00686630681772506;
              }
            } else {
              result[0] += 0.02850005842786903;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.0039010386956722237;
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.018270379819844605;
                  } else {
                    result[0] += 0.0017115693437391236;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.020188296034874247;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                        result[0] += -0.0826171088110737;
                      } else {
                        result[0] += -0.014412712381962366;
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.01323088037744172;
                        } else {
                          result[0] += -0.039275941658541154;
                        }
                      } else {
                        result[0] += -0.036896587317280946;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.017083320942070015;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.06411429350163013;
                    } else {
                      result[0] += -0.014875924337831066;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.007812325106782318;
                  } else {
                    result[0] += 0.009259920721603171;
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    result[0] += 0.027707014946408395;
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
                      result[0] += 0.03767954887040589;
                    } else {
                      result[0] += -0.06530601678050657;
                    }
                  }
                }
              } else {
                result[0] += 0.0033799694009166814;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                result[0] += -0.051198076569072985;
              } else {
                result[0] += 0.09755170771466717;
              }
            } else {
              result[0] += -0.09008371266807177;
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
                  result[0] += 0.004631204352654505;
                } else {
                  result[0] += 0.09401745578734522;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += -0.03097376204281447;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.0972093201316836;
                  } else {
                    result[0] += 0.07412763353756643;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                result[0] += 0.01694429671561375;
              } else {
                result[0] += -0.09035690237851053;
              }
            }
          }
        }
      } else {
        result[0] += -0.01316454367246716;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.032563480735055716;
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.08555670305370879;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.0874079131574872;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.054721733842046254;
                } else {
                  result[0] += -0.05157383726085344;
                }
              } else {
                result[0] += -0.043912787913107404;
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.06644226610060304;
                  } else {
                    result[0] += -0.07581675518575987;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)232.0000000000000284) ) ) {
                    result[0] += -0.06266725348831523;
                  } else {
                    result[0] += 0.0515821528665705;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.004200399775464099;
                } else {
                  result[0] += 0.03364570174548119;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.06740312609455067;
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.04853305676869299;
          } else {
            result[0] += 0.030628355715310668;
          }
        }
      } else {
        result[0] += 0.053721003320895036;
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.05315554855241353;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.05303162700923167;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.06079686844345037;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.018636892386726993;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.08295939874886174;
                } else {
                  result[0] += 0.005893333704937178;
                }
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += 0.01532328786601784;
                    } else {
                      result[0] += -0.04062831081576468;
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += -0.014129731744602338;
                    } else {
                      result[0] += -0.06271818697329769;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.020395072652644355;
                  } else {
                    result[0] += -0.01355591516741753;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.0010339263917630217;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += -0.11853794643924387;
          } else {
            result[0] += 0.09387681557956318;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.04006882255093241;
                      } else {
                        result[0] += 0.0013713950160097496;
                      }
                    } else {
                      result[0] += 0.010143259070741296;
                    }
                  } else {
                    result[0] += -0.048257153291439314;
                  }
                } else {
                  result[0] += 0.0582702087476332;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.09036704898103826;
                } else {
                  result[0] += -0.07597118769589714;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += 0.0812914794484957;
                      } else {
                        result[0] += -0.0765248958144351;
                      }
                    } else {
                      result[0] += -0.0001821397414587051;
                    }
                  } else {
                    result[0] += -0.13490440601433876;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                    result[0] += -0.11640918469084399;
                  } else {
                    result[0] += -0.041522509041064054;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.020247818691498953;
                } else {
                  result[0] += 0.04244369709721584;
                }
              }
            }
          } else {
            result[0] += -0.09321716466760065;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
                  result[0] += -0.06155865696583745;
                } else {
                  result[0] += 0.06354879345359266;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.05985834864457262;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.00954571499344626;
                  } else {
                    result[0] += 0.07595074307658392;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                    result[0] += -0.052397752453715654;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += -0.06699393534353296;
                      } else {
                        result[0] += -0.008571489996229203;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                        result[0] += -0.013420326903139724;
                      } else {
                        result[0] += 0.044797215682717495;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += 0.03385838438271935;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                          result[0] += 0.004800251785692394;
                        } else {
                          result[0] += -0.14309343805323632;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.00964519037784935;
                      } else {
                        result[0] += 0.06432046259955668;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
                      result[0] += 0.0007410409840931786;
                    } else {
                      result[0] += -0.06463575879244003;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.07193866896463484;
                  } else {
                    result[0] += 0.022130307827267767;
                  }
                } else {
                  result[0] += 0.04043215394611076;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.004674338702338186;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.027272982512426033;
                  } else {
                    result[0] += -0.010604356915135872;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.04438431050049532;
                    } else {
                      result[0] += 0.0004584701439776031;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.005655368273084629;
                    } else {
                      result[0] += -0.07808374591737399;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                    result[0] += -0.007459720971233218;
                  } else {
                    result[0] += 0.034618069645037684;
                  }
                } else {
                  result[0] += -0.06666169394886534;
                }
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                    result[0] += 0.04093200899169189;
                  } else {
                    result[0] += -0.07053660874403254;
                  }
                } else {
                  result[0] += -0.09760966904194078;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.06685416004743562;
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.08305370116062029;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.08963097567914996;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.16078476106237172;
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.05355613239758165;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.01770932720436087;
                  } else {
                    result[0] += -0.05304345592039034;
                  }
                } else {
                  result[0] += 0.02845078539248208;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      result[0] += 0.034727950929954836;
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.050594218892375004;
      } else {
        result[0] += 0.004946123185747242;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.00047531306913734217;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.04298033491503484;
            } else {
              result[0] += -0.08456152587287527;
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += -0.020737402076633858;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += 0.012519549760930272;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.03331066673424122;
                  } else {
                    result[0] += 0.08948223726107579;
                  }
                }
              }
            } else {
              result[0] += -0.03645031385249831;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
          result[0] += 0.019366212886352194;
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += 0.004249497059885242;
            } else {
              result[0] += -0.06179962863002192;
            }
          } else {
            result[0] += 0.011891544809235492;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.08155846595764249) ) ) {
                      result[0] += -0.09099900838206071;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.11130205810345317;
                      } else {
                        result[0] += -0.05586712819976619;
                      }
                    }
                  } else {
                    result[0] += 0.012553826002217095;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += 0.10952048160913351;
                      } else {
                        result[0] += -0.030507912594466086;
                      }
                    } else {
                      result[0] += -0.08513951340215159;
                    }
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += 0.00914924920426036;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.585059762001038042) ) ) {
                            if ( LIKELY(  (data[47].missing != -1) && (data[47].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                result[0] += -0.06297940011001575;
                              } else {
                                result[0] += -0.017972821697122988;
                              }
                            } else {
                              result[0] += 0.016762336346431202;
                            }
                          } else {
                            result[0] += 0.03171838787060987;
                          }
                        } else {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                            result[0] += 0.019364678736112976;
                          } else {
                            result[0] += -0.061510178090297345;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.03201916225829172;
                    }
                  }
                }
              } else {
                result[0] += -0.04161491593309322;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.04915290926120846;
                  } else {
                    result[0] += -0.07435852836790634;
                  }
                } else {
                  result[0] += -0.03861880069058779;
                }
              } else {
                result[0] += 0.07097718723979593;
              }
            }
          } else {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.0729654569427181;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                        result[0] += 0.07037889162325751;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += 0.05245799590909109;
                        } else {
                          result[0] += -0.02838355663892188;
                        }
                      }
                    } else {
                      result[0] += 0.10469466797392053;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.021375295024217485;
                  } else {
                    result[0] += -0.07911189908858379;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                      result[0] += 0.09370877512089215;
                    } else {
                      result[0] += 0.0372479759762255;
                    }
                  } else {
                    result[0] += 0.014246819407617087;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.005153545553852217;
                  } else {
                    result[0] += -0.08787197363368526;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += -0.006306093378411173;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += -0.0779491096798526;
                  } else {
                    result[0] += -0.024186877317629914;
                  }
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                      result[0] += 0.0013697474395082232;
                    } else {
                      result[0] += 0.029618219083494397;
                    }
                  } else {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.10666956924781312;
                      } else {
                        result[0] += -0.016586849493424077;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                        result[0] += 0.0058744517576401305;
                      } else {
                        result[0] += -0.016604517690594907;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                    result[0] += -0.05669896666109965;
                  } else {
                    result[0] += 0.028830394208819707;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.011785299512561579;
              } else {
                result[0] += -0.07733876137975321;
              }
            } else {
              result[0] += 0.013803568142752546;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
              result[0] += 0.006523034044890498;
            } else {
              result[0] += 0.043599479194943586;
            }
          }
        }
      } else {
        result[0] += -0.09683368555925498;
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
      result[0] += -0.02910693079489575;
    } else {
      result[0] += 0.015812933573327262;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
          result[0] += -5.542356870424639e-05;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.43742394447326749) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.008184965581068508;
                } else {
                  result[0] += 0.06508409093893221;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                    result[0] += 0.008234711470407285;
                  } else {
                    result[0] += -0.051100186395797546;
                  }
                } else {
                  result[0] += 0.0247462872385919;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.09183686410414649;
              } else {
                result[0] += 0.014054075590928604;
              }
            }
          } else {
            result[0] += -0.04575131056898528;
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
          result[0] += 0.13940331902403721;
        } else {
          result[0] += -0.11197647975461918;
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
                        result[0] += -0.010211435372865672;
                      } else {
                        result[0] += -0.04782043243485661;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.09515759340796175;
                      } else {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.05188577486506695;
                        } else {
                          result[0] += -0.004469121256176192;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                      result[0] += 0.010457614486439622;
                    } else {
                      result[0] += -0.07661234063295974;
                    }
                  }
                } else {
                  result[0] += 0.004283706818748099;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += -0.034105615593737426;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                      result[0] += 0.04728846537285478;
                    } else {
                      result[0] += -0.07498789703508615;
                    }
                  }
                } else {
                  result[0] += 0.05906644460311025;
                }
              }
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.08069349670160597;
                    } else {
                      result[0] += 0.07837683565891262;
                    }
                  } else {
                    result[0] += -0.02322098729030805;
                  }
                } else {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.05868701139119486;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += 0.02151028787398484;
                      } else {
                        result[0] += 0.09845268794424988;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += 0.023136590331590268;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.021290660848281115;
                      } else {
                        result[0] += -0.08610592160538337;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.042435407638550693) ) ) {
                    result[0] += 0.011834206030412771;
                  } else {
                    result[0] += -0.06875718452335369;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.962127923965454546) ) ) {
                    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += -0.04490319296217398;
                    } else {
                      result[0] += 0.014437384388441974;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                            result[0] += -0.17218188257462466;
                          } else {
                            result[0] += 0.014732565331213438;
                          }
                        } else {
                          result[0] += -0.012291927355879043;
                        }
                      } else {
                        result[0] += -0.08958160387849534;
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += 0.012184669482039175;
                        } else {
                          result[0] += -0.021192024028516;
                        }
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.0005948652999128603;
                        } else {
                          result[0] += -0.035118712400138086;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.04524737215397338;
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
              result[0] += -0.05357140042995513;
            } else {
              result[0] += -0.0002864878322250199;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.032210794044337555;
            } else {
              result[0] += -0.009727366813714139;
            }
          }
        }
      } else {
        result[0] += -0.09662082448926235;
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += -0.015755223873470024;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.047911355208551165;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.6028149597243004;
                } else {
                  result[0] += -0.03707485524696318;
                }
              }
            } else {
              result[0] += -0.15511270451468803;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += -0.03186540866979754;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += -0.1095087526903845;
            } else {
              result[0] += 0.08837228770783039;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.05002601437689341;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.026692019785773475;
          } else {
            result[0] += 0.028478246621914234;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        result[0] += 0.037879720805222866;
      } else {
        result[0] += 0.00879930407061004;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                result[0] += 0.07022980497888075;
              } else {
                result[0] += 0.010609126340033491;
              }
            } else {
              result[0] += -0.05836999964214552;
            }
          } else {
            result[0] += 0.05793420405229769;
          }
        } else {
          result[0] += 0.0014165422981554708;
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
            if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += -0.0013513049404758107;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                        result[0] += -0.025710856104221066;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                            result[0] += -0.042802140845791786;
                          } else {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                              result[0] += -0.016349795797374483;
                            } else {
                              result[0] += 0.0450349644042918;
                            }
                          }
                        } else {
                          result[0] += -0.0278645006896104;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.0652116805525948;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.01053571701049982) ) ) {
                    result[0] += 0.0021549944108530277;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.09021948626284296;
                      } else {
                        result[0] += -0.012069558183458458;
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.08189665532580707;
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.03819249906004585;
                        } else {
                          result[0] += 0.022059789737658857;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.0949927152867802;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                          result[0] += -0.031098939586417365;
                        } else {
                          result[0] += 0.062362695643825516;
                        }
                      } else {
                        result[0] += -0.08234871646093261;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.05913919278553417;
                      } else {
                        result[0] += 0.015016361379125279;
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.019647048015161758;
                      } else {
                        result[0] += -0.058708361994449;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.058919551578897916;
                      } else {
                        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.027604845195764446;
                        } else {
                          result[0] += 0.023320579595091477;
                        }
                      }
                    } else {
                      result[0] += -0.06672199618844113;
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += -0.005409820816029265;
                        } else {
                          result[0] += -0.10266657666419404;
                        }
                      } else {
                        result[0] += 0.014500181251964051;
                      }
                    } else {
                      result[0] += -0.00034263388931949154;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                  result[0] += 0.04448921504019588;
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                      result[0] += 0.02648519645582293;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.05283861393088044;
                      } else {
                        result[0] += 0.027914399269188168;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += -0.009075776469859853;
                      } else {
                        result[0] += 0.1053787151014986;
                      }
                    } else {
                      if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                          result[0] += -0.1253691458416227;
                        } else {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                            result[0] += -0.08260523076073914;
                          } else {
                            result[0] += 0.01765613805443346;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                          result[0] += 0.028782836410637554;
                        } else {
                          result[0] += -0.036269246160861156;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += 0.020692013747995255;
                  } else {
                    result[0] += -0.075241926123641;
                  }
                } else {
                  result[0] += -0.05677826088313769;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += -0.001492825752137837;
            } else {
              result[0] += -0.06260956514112102;
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
            result[0] += 0.013775646769753055;
          } else {
            result[0] += -0.07139007860467489;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
          result[0] += -0.05845965206743321;
        } else {
          result[0] += 0.0008237780866434864;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.03527368320723807;
            } else {
              result[0] += -0.09741687853267095;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += 0.06241064860004645;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += -0.06907156644253827;
              } else {
                result[0] += 0.025964542553820965;
              }
            }
          }
        } else {
          result[0] += 0.052992165981444544;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
      result[0] += -0.017060146032827232;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += -0.03442452510270721;
      } else {
        result[0] += 0.01813292966847676;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.010186767606776162;
          } else {
            result[0] += 0.0557467037671054;
          }
        } else {
          result[0] += 0.0011388289134024638;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.007201243109112396;
                } else {
                  result[0] += -0.04343623887903759;
                }
              } else {
                result[0] += -0.07871396652139101;
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                result[0] += 0.0012566519150780074;
              } else {
                result[0] += -0.026852973614160544;
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.849175214767456943) ) ) {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += 0.08974227893509128;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                          result[0] += 0.05350936275008109;
                        } else {
                          result[0] += -0.06369885958009482;
                        }
                      }
                    } else {
                      result[0] += 0.06313806549466937;
                    }
                  } else {
                    result[0] += -0.019774751018100903;
                  }
                } else {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.05982848019181197;
                  } else {
                    result[0] += 0.0026403753699146687;
                  }
                }
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                    result[0] += 0.05594209834622313;
                  } else {
                    result[0] += -0.061993793823290747;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += -0.2099397871086521;
                  } else {
                    result[0] += 0.0382640004321684;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                result[0] += -0.09531842607801737;
              } else {
                result[0] += 0.03431271577142934;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += -0.09107175914147259;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
                            result[0] += 0.04220410829512885;
                          } else {
                            result[0] += -0.07450153950416287;
                          }
                        } else {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.030759394537200654;
                          } else {
                            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                              result[0] += 0.006696682762886407;
                            } else {
                              result[0] += 0.11886236265470884;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.02640153970339268;
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.1446032690247742;
                            } else {
                              result[0] += 0.044644356659036474;
                            }
                          }
                        } else {
                          result[0] += -0.037142557717509;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                          result[0] += 0.05583147331941527;
                        } else {
                          result[0] += -0.0017663040663791246;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                          result[0] += 0.021545562172554024;
                        } else {
                          result[0] += 0.10466952685245395;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += -0.0758781982549343;
                      } else {
                        result[0] += 0.004286436552478295;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                    result[0] += -0.008751596736374528;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                      result[0] += -0.02891801902823481;
                    } else {
                      result[0] += 0.04848150050964014;
                    }
                  }
                }
              } else {
                result[0] += -0.0581651361493106;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.06669078669704757;
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                      result[0] += 0.04711449197211642;
                    } else {
                      result[0] += -0.10883289105725197;
                    }
                  }
                } else {
                  result[0] += -0.08303611420898618;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                      result[0] += 0.03610547156889607;
                    } else {
                      result[0] += -0.102577606857845;
                    }
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                      result[0] += 0.21854071607440662;
                    } else {
                      result[0] += -0.037682947500983235;
                    }
                  }
                } else {
                  result[0] += -0.07738268875086601;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
              result[0] += -0.04381972101087524;
            } else {
              result[0] += 0.046425886097014514;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
        result[0] += -0.017331902933164438;
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
              result[0] += 0.03936730691651544;
            } else {
              result[0] += -0.062278350027184196;
            }
          } else {
            result[0] += 0.04581105640725526;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
            result[0] += -0.027695073371601056;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              result[0] += 0.08298857392313629;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                result[0] += 0.011756668765194163;
              } else {
                result[0] += 0.05336686273499258;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
      result[0] += -0.07140903124766423;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        result[0] += -0.10115054197261096;
      } else {
        result[0] += 0.013512763999692113;
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
        result[0] += 0.00024359873223858336;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.01891749514593548;
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.08249954669602236;
          } else {
            result[0] += 0.007050478122867897;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.008407717521385746;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                    result[0] += -0.00843542498305434;
                  } else {
                    result[0] += -0.08699350892630381;
                  }
                } else {
                  result[0] += -0.014066997591311803;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                    result[0] += -0.04544250282736374;
                  } else {
                    result[0] += 0.09786720514701211;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.008911256271364496;
                    } else {
                      result[0] += 0.013797898570404374;
                    }
                  } else {
                    result[0] += -0.08265605928640984;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.021814427351830246;
                  } else {
                    result[0] += 0.004917384880381478;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                          result[0] += 0.01178634951810143;
                        } else {
                          result[0] += 0.08078204133296218;
                        }
                      } else {
                        result[0] += -0.034431684645483004;
                      }
                    } else {
                      result[0] += -0.023997562705915555;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.01394551774960149;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                        result[0] += -0.05952094059634351;
                      } else {
                        result[0] += 0.1266379807231213;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.07392676697682647;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.09881339855295102;
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.05563420636751992;
                        } else {
                          result[0] += 0.029059152181092337;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                          result[0] += 0.042418561203297754;
                        } else {
                          result[0] += -0.04428041719175775;
                        }
                      }
                    } else {
                      result[0] += 0.031415614594684824;
                    }
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.06380209301212927;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                              result[0] += 0.08695938131437547;
                            } else {
                              result[0] += 0.0017534213593839937;
                            }
                          } else {
                            result[0] += -0.06672995604145543;
                          }
                        } else {
                          result[0] += -0.0457392290101243;
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                            result[0] += -0.06741642822537171;
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                              result[0] += -0.029410953581992822;
                            } else {
                              result[0] += 0.07202839014727104;
                            }
                          }
                        } else {
                          result[0] += 0.00800736873186205;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.0019323448486793801;
            }
          }
        } else {
          result[0] += -0.09610699801327519;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += -0.05480037184305112;
          } else {
            result[0] += 0.0898114622606261;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += 0.13530328466978633;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.04809766002227869;
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0732078871992815;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.013304284340293477;
                  } else {
                    result[0] += -0.052169072574395396;
                  }
                } else {
                  result[0] += 0.02485079503349967;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += 0.011572513010154772;
      } else {
        result[0] += 0.046252998754120866;
      }
    } else {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.008911107440622087;
                } else {
                  result[0] += -0.11916307672474012;
                }
              } else {
                result[0] += 0.11251631845738992;
              }
            } else {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += -0.10810934153198903;
              } else {
                result[0] += 0.02176143694224886;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += 0.05653450265429308;
            } else {
              result[0] += -0.06702487889958585;
            }
          }
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.09741648898449767;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
              result[0] += -0.06602087912922198;
            } else {
              result[0] += 0.08226425008924364;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.09514836452393013;
          } else {
            result[0] += -0.01662226128894554;
          }
        } else {
          result[0] += 0.0017404294870508197;
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.0004223087621174994;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -0.07866506523362204;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
              result[0] += -0.01976730322776729;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.025576965091687234;
                  } else {
                    result[0] += -0.07651216750302937;
                  }
                } else {
                  result[0] += 0.039993905904230014;
                }
              } else {
                result[0] += 0.06113229908927991;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.00970505155550362;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += -0.08305464078910607;
              } else {
                result[0] += -0.016903514635811395;
              }
            }
          } else {
            result[0] += -0.06610345043713128;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.10970263434804468;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.34969997406006037) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.16129454775618113;
                  } else {
                    result[0] += -0.006920865960661989;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.10412316985148402;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.10020112720701634;
                      } else {
                        result[0] += 0.02809790749167378;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.65882921218872248) ) ) {
                          result[0] += -0.02635095981514593;
                        } else {
                          result[0] += 0.06981520724940309;
                        }
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.935600519180298074) ) ) {
                          result[0] += -0.01494619485358565;
                        } else {
                          result[0] += 0.10057850466782087;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.869339942932130683) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                        result[0] += -0.011450673362733222;
                      } else {
                        result[0] += 0.16962630256997585;
                      }
                    } else {
                      result[0] += -0.07692556925492182;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.022074957712639814;
                      } else {
                        result[0] += 0.10879386401507773;
                      }
                    } else {
                      result[0] += -0.08077993838987212;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                    result[0] += 0.00843732617259142;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.14565390225834815;
                    } else {
                      result[0] += -0.11323702012085318;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += -0.14489718484661085;
                  } else {
                    result[0] += -0.02243271798626219;
                  }
                } else {
                  result[0] += 0.04028738798307565;
                }
              } else {
                result[0] += -0.05727238613306698;
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.030717673340844;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.016358396251838773;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                        result[0] += -0.014782089712789187;
                      } else {
                        result[0] += 0.061645316688159826;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.589026927947998269) ) ) {
                      result[0] += -0.001591251212900759;
                    } else {
                      result[0] += -0.11108664427375495;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY(  (data[45].missing != -1) && (data[45].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
                      result[0] += -0.0006782741208974381;
                    } else {
                      result[0] += 0.13707765901794164;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                        result[0] += -0.06431385902227028;
                      } else {
                        result[0] += -0.002779668619943134;
                      }
                    } else {
                      result[0] += -0.10066414761823268;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.06613002229201283;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                        result[0] += -0.10153519256008639;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.024840572454216475;
                        } else {
                          result[0] += -0.11690477115834896;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04255271453723381;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.07732810134493245;
          } else {
            result[0] += -0.042028007924276335;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.3915819056337525;
          } else {
            result[0] += -0.015981227496987875;
          }
        }
      } else {
        result[0] += 0.1528391558572078;
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        result[0] += 0.03676757080318054;
      } else {
        result[0] += 0.00014401493498272485;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.88435244560241788) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
            result[0] += 0.0006992674157505565;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.695749998092652255) ) ) {
              result[0] += -0.06869724779866633;
            } else {
              result[0] += 0.004625469872425333;
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
            result[0] += 0.10441493562879618;
          } else {
            result[0] += -0.07702401974234838;
          }
        }
      } else {
        result[0] += 0.01688425431831055;
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.025458797107007228;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.978102684020996982) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.028376869274281288;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.076962471008301669) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.515218973159790483) ) ) {
                        result[0] += 0.04210730134587387;
                      } else {
                        result[0] += -0.008748198718243633;
                      }
                    } else {
                      result[0] += -0.07588758016637893;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.34969997406006037) ) ) {
                    result[0] += 0.056041695702934546;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.16186278650687239;
                    } else {
                      result[0] += -0.054625443486538416;
                    }
                  }
                }
              } else {
                result[0] += 0.002812970667048434;
              }
            } else {
              result[0] += 0.03338939691477606;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.376888275146486151) ) ) {
            result[0] += -0.005764765061365221;
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
              result[0] += 0.034921908922791525;
            } else {
              result[0] += -0.08298415891604255;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)14.50000000000000178) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                            result[0] += 0.0031234888373468384;
                          } else {
                            result[0] += -0.07282571737165786;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.790835380554201) ) ) {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.06894607219447178;
                              } else {
                                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                                  result[0] += -0.039186582969021604;
                                } else {
                                  result[0] += 0.010651770106117573;
                                }
                              }
                            } else {
                              result[0] += 0.09024068778003401;
                            }
                          } else {
                            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                                result[0] += -0.012026390853141965;
                              } else {
                                result[0] += 0.1445650685506233;
                              }
                            } else {
                              result[0] += -0.049146992737704884;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.003039794826548808;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.3070559501647967) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.11056157349723775;
                            } else {
                              result[0] += 0.006633305997421832;
                            }
                          } else {
                            result[0] += -0.10116471275357586;
                          }
                        } else {
                          result[0] += -0.05398971527893884;
                        }
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                          result[0] += 0.009150635241224829;
                        } else {
                          result[0] += -0.027492221777228487;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                      result[0] += -0.008113638222236222;
                    } else {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.06852357840143239;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.08166739010581431;
                        } else {
                          result[0] += 0.012150906339365983;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.07439206364340961;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                    result[0] += 0.049477645220847445;
                  } else {
                    result[0] += 0.01682210610184537;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                                result[0] += -0.07648799393619563;
                              } else {
                                result[0] += 0.0985194615170451;
                              }
                            } else {
                              result[0] += 0.46650329844706706;
                            }
                          } else {
                            result[0] += -0.035737035163608065;
                          }
                        } else {
                          result[0] += -0.07004905214059436;
                        }
                      } else {
                        result[0] += -0.059693585278960286;
                      }
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
                        result[0] += -0.06312089831559;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                            result[0] += -0.09843283795817508;
                          } else {
                            result[0] += 0.0884006241792328;
                          }
                        } else {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                            result[0] += 0.009605334773910755;
                          } else {
                            result[0] += -0.0694482918585201;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.08220882646255283;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.06028245394589632;
              } else {
                result[0] += -0.008228481379839316;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.88435244560241788) ) ) {
              result[0] += 0.017167725668792027;
            } else {
              result[0] += -0.10570024506222658;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
            result[0] += -0.18113370997657816;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.11152004226984057;
                } else {
                  result[0] += -0.020843254516058445;
                }
              } else {
                result[0] += -0.08500227272304474;
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += 0.008146312807514695;
                  } else {
                    result[0] += -0.138682814266867;
                  }
                } else {
                  result[0] += 0.08895563140016385;
                }
              } else {
                result[0] += 0.026957725821389018;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        result[0] += 0.0632915157064116;
      } else {
        result[0] += -0.027226186791697332;
      }
    }
  } else {
    result[0] += 0.01048933907810892;
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
          if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
                result[0] += 0.0026365456290261538;
              } else {
                result[0] += -0.030716944420901976;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
                result[0] += 0.004948153886340635;
              } else {
                result[0] += 0.039305647348325944;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.0008875891165407142;
                } else {
                  result[0] += 0.07816960598360742;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += 0.009811533982659396;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.014974448190534681;
                    } else {
                      result[0] += 0.03822526223176635;
                    }
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                        result[0] += -0.024796588079005602;
                      } else {
                        result[0] += 0.009274701368613204;
                      }
                    } else {
                      result[0] += -0.07586512147430641;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.3450207320187779;
                  } else {
                    result[0] += 0.044637154836949876;
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.05678766370995162;
                  } else {
                    result[0] += 0.015747862314944083;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05715419752328983;
                } else {
                  result[0] += 0.0018198208757690204;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
            result[0] += -0.040688393933930554;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                result[0] += -0.03636177967268499;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0633054110591043;
                } else {
                  result[0] += -0.09351003263057989;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.03541666235946083;
              } else {
                result[0] += -0.08971965747573982;
              }
            }
          }
        }
      } else {
        result[0] += -0.01970883536683832;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.0676220408178315;
              } else {
                result[0] += -0.03388919681114031;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.07885621363733801;
              } else {
                result[0] += -0.022604093700988613;
              }
            }
          } else {
            result[0] += 0.05036895739401923;
          }
        } else {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.1284973812066732;
          } else {
            result[0] += -0.004162799944938015;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.033544585028407646;
            } else {
              result[0] += -0.10447457460102201;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.0703425025118382;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                result[0] += 0.09472997867763212;
              } else {
                result[0] += -0.016031786881317196;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            result[0] += -0.0456871834894586;
          } else {
            result[0] += 0.05891562289642741;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += 0.0626636390004059;
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += 0.05200546925723518;
          } else {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.10131438749028321;
            } else {
              result[0] += 0.016142277913430782;
            }
          }
        } else {
          result[0] += -0.10105386783771758;
        }
      }
    } else {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.03936922573296132;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.03023747512743182;
              } else {
                result[0] += -0.08882552779138457;
              }
            } else {
              result[0] += 0.003180385376360047;
            }
          }
        } else {
          if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.027482268874729363;
            } else {
              result[0] += 0.02797227056475285;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
              result[0] += -0.10014972694324634;
            } else {
              result[0] += 0.0034398945657332734;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.03170185454844045;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.007157102518153449;
            } else {
              result[0] += -0.05993429785622944;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.04783278684507743;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03232498925262593;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.08843840820375093;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.10454121841800822;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.029890486020610164;
                        } else {
                          result[0] += 1.249294065769524;
                        }
                      }
                    } else {
                      result[0] += -0.10223625727531693;
                    }
                  }
                } else {
                  result[0] += 0.13216836029207613;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.023118689911890422;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                result[0] += -0.001053613518426124;
              } else {
                result[0] += 0.02674457034363749;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
              result[0] += 0.004896405955272506;
            } else {
              result[0] += 0.03897948372373178;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.007480413526533708;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.02896505824038613;
            } else {
              result[0] += -0.13257103495083927;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          result[0] += -0.002602356655009554;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.005837338467648779;
            } else {
              result[0] += -0.08737872042782495;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.580392837524414951) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.03913296262982902;
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0732976429390008;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.06449976636788883;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                        result[0] += -0.04779775592246018;
                      } else {
                        result[0] += 0.02049948739388777;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0762585124083396;
              }
            } else {
              result[0] += -0.04368936431939066;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
        result[0] += -0.005460465986557183;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.027745978845010868;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += -0.029311218876073283;
            } else {
              result[0] += 0.12192982379344341;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.06282444003347987;
          } else {
            result[0] += 0.05357128168625474;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.07886634354035074;
      } else {
        result[0] += 0.020547986152750462;
      }
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.08871694429074885;
          } else {
            result[0] += 0.019495846449082546;
          }
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.055751901123358316;
            } else {
              result[0] += -0.0779119113855109;
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
                result[0] += 0.030944730161882917;
              } else {
                result[0] += -0.04143959649748674;
              }
            } else {
              result[0] += 0.044549764871760315;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += 0.10249587294604126;
          } else {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
              result[0] += -0.08283024132329433;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                result[0] += 0.06728020256613373;
              } else {
                result[0] += -0.09781240432975268;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.06158579450029003;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.06327233523475491;
                  } else {
                    result[0] += -0.0353729259072274;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
                      result[0] += -0.09813258416618995;
                    } else {
                      result[0] += -0.020192140311923158;
                    }
                  } else {
                    result[0] += 0.023368083317206096;
                  }
                }
              } else {
                if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                    result[0] += -0.025593174160492206;
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                      result[0] += 0.032167456809913605;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                        result[0] += 0.06211574601137457;
                      } else {
                        result[0] += -0.07920387646295425;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                    result[0] += -0.10950634123903431;
                  } else {
                    result[0] += 0.003025134182326436;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                  result[0] += -0.03902584475395671;
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.04198549176214636;
                  } else {
                    result[0] += -0.10524896495785896;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.002424837866553798;
                  } else {
                    result[0] += -0.06128311379695093;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.07551300221424656;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            result[0] += -0.02253788381674418;
                          } else {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.029437273992347412;
                            } else {
                              result[0] += 0.6088639679699142;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                              result[0] += -0.10087922721650366;
                            } else {
                              result[0] += 0.0986126615205885;
                            }
                          } else {
                            result[0] += -0.10059296836213179;
                          }
                        }
                      } else {
                        result[0] += 0.1351349642597552;
                      }
                    }
                  } else {
                    result[0] += -0.03520414197882194;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.769779443740845171) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.035357315447629156;
              } else {
                result[0] += 0.003770155642134466;
              }
            } else {
              result[0] += -0.06224448616169927;
            }
          } else {
            result[0] += 0.08177065768381442;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.01345280557000011;
            } else {
              result[0] += 0.05747521129192813;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.284418344497681552) ) ) {
              result[0] += -0.0007697814522508997;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                  result[0] += -0.1864584201098855;
                } else {
                  result[0] += -0.03920418542987975;
                }
              } else {
                result[0] += 0.011690615928321749;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)5.000000000000000888) ) ) {
            result[0] += -0.0023647662949457626;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += -0.014352700355490015;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.012522199338289523;
                    } else {
                      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                        result[0] += 0.05483091890510147;
                      } else {
                        result[0] += 0.16690830448380725;
                      }
                    }
                  }
                } else {
                  result[0] += -0.10207844812846636;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.07942668840937367;
                  } else {
                    result[0] += 0.017437393032962708;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.06383862799841013;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.04249966834199867;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += 0.07481714856897315;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                            result[0] += -0.10013088951825547;
                          } else {
                            result[0] += 0.048229791506156555;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                      result[0] += -0.11043225224400362;
                    } else {
                      result[0] += -0.00825702513592212;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.07968285651393553;
                  } else {
                    result[0] += -0.11416958312950687;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += -0.016655505829526077;
                    } else {
                      result[0] += 0.15355012283039537;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.0781804541705453;
                    } else {
                      result[0] += -0.05872408960246847;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += -0.08101903817446302;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.040926622612584326;
                    } else {
                      result[0] += -0.030771409287039392;
                    }
                  }
                } else {
                  result[0] += -0.08471573510340182;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
            result[0] += -0.03764674309611428;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                result[0] += -0.03539113317372442;
              } else {
                result[0] += 0.028443337440614693;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.03288794851879851;
              } else {
                result[0] += -0.08321922507218597;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
        result[0] += -0.0130678875417145;
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              result[0] += 0.04025187955200488;
            } else {
              result[0] += -0.05861572762854653;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.06315545575858565;
            } else {
              result[0] += -0.09575268135307935;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
            result[0] += -0.02471163401428679;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.07651496968461491;
              } else {
                result[0] += -0.05472153069578217;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += -0.009393835987999012;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.018215483329148328;
                    } else {
                      result[0] += 0.10714839503159759;
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += -0.07193408867865755;
                    } else {
                      result[0] += 0.02393397618060276;
                    }
                  }
                } else {
                  result[0] += 0.05451028786015288;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          result[0] += 0.02530706026709781;
        } else {
          result[0] += -0.08213875110967368;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.135127481322784;
        } else {
          result[0] += 0.00042290749196167537;
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.049257751077675875;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.03943822898970822;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.05912433861180302;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.015173166365090265;
            } else {
              result[0] += 0.0006688015351453575;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
          result[0] += -0.0016185481353720738;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.006625875444641109;
          } else {
            result[0] += 0.019072132312812795;
          }
        }
      } else {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.0383294040089417;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.008339228252411825;
          } else {
            result[0] += -0.038792268300032046;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.02763858816816522;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
              result[0] += 0.09789053942449427;
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += -0.08610448760267045;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += -0.07817857536187828;
                } else {
                  result[0] += 0.05004703548091615;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06899832842236148;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.01288357898874987;
              } else {
                result[0] += 0.052061078287145825;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.010625153846357475;
        } else {
          result[0] += 0.04286205967793613;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.00749943936106618;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += 0.06139669766648401;
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                result[0] += 0.03101555629888582;
              } else {
                result[0] += -0.06499538837573916;
              }
            }
          }
        } else {
          result[0] += -0.08566279929966;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.13072355774613417;
        } else {
          result[0] += 0.00970872972469622;
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.07467518060918425;
        } else {
          result[0] += 0.020841849690315987;
        }
      } else {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                result[0] += 0.04743999654820308;
              } else {
                result[0] += -0.0815213179926968;
              }
            } else {
              if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += 0.03167427358674956;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                        result[0] += -0.02775002171973992;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.17598468385464638;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 4.517057765249917;
                            } else {
                              result[0] += 0.7411457453738909;
                            }
                          } else {
                            result[0] += 0.6622631870499308;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.08406980370740931;
                    }
                  } else {
                    result[0] += -0.14327316194672643;
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 2.06233496372974;
                            } else {
                              result[0] += 0.05754968232120827;
                            }
                          } else {
                            result[0] += 3.527009825486253;
                          }
                        } else {
                          result[0] += 0.2582137283932426;
                        }
                      } else {
                        result[0] += 0.01686547487066407;
                      }
                    } else {
                      result[0] += -0.03604806093030201;
                    }
                  } else {
                    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.045093244705973964;
                          } else {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                              result[0] += -0.05117996011723759;
                            } else {
                              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                                  result[0] += -0.03236930670988272;
                                } else {
                                  result[0] += 0.1498809403448049;
                                }
                              } else {
                                result[0] += 0.0832679020566454;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.059353604477132275;
                        }
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                          result[0] += 0.04919397553990286;
                        } else {
                          result[0] += 0.12663009256775104;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                        result[0] += -0.04051050905480454;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                          result[0] += 0.09603688413751932;
                        } else {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += -0.041745490951922995;
                          } else {
                            result[0] += 0.026817162964545973;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                    result[0] += 0.022081710303517764;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += -0.0026803290847360753;
                    } else {
                      result[0] += -0.055798964620714764;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.040100029575869986;
          }
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
              result[0] += 0.000689100973429529;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.08551731402740798;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.03057143221048339;
                  } else {
                    result[0] += 0.039356261908709125;
                  }
                } else {
                  result[0] += 0.10095155084706325;
                }
              }
            }
          } else {
            result[0] += 0.0035804096161086403;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.0070632593962748;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                  result[0] += 0.12595978314091197;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.058455410470069524;
                  } else {
                    result[0] += -0.2001176283180488;
                  }
                }
              } else {
                result[0] += 0.10862027171438755;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.005853766464428333;
            } else {
              result[0] += 0.05160608771765551;
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.01181635357438664;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.790835380554201) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.842307567596437323) ) ) {
                result[0] += -0.0175944687464844;
              } else {
                result[0] += -0.06415552173664381;
              }
            } else {
              result[0] += -0.005729537848889137;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.002295641517297174;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.008871060764494646;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.04572711040176695;
              } else {
                result[0] += -0.03507665253937489;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.023153214962400234;
                  } else {
                    result[0] += -0.08949354275418556;
                  }
                } else {
                  result[0] += 0.08531992746865012;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.431901693344116655) ) ) {
                    result[0] += -0.13542296290928516;
                  } else {
                    result[0] += -0.017808113936077348;
                  }
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
                    result[0] += 0.00869999321712828;
                  } else {
                    result[0] += -0.03726906128426303;
                  }
                }
              }
            } else {
              result[0] += -0.08774620348045996;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.07624601788160235;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.10329914989453529;
                    } else {
                      result[0] += -0.0023403585825594398;
                    }
                  } else {
                    result[0] += 0.03979412078938415;
                  }
                } else {
                  result[0] += -0.025991048067592143;
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.935600519180298074) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                        result[0] += 0.026250825951602788;
                      } else {
                        result[0] += -0.06535134485364578;
                      }
                    } else {
                      result[0] += 0.12580314081894303;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.585059762001038042) ) ) {
                        result[0] += -0.04963148517625185;
                      } else {
                        result[0] += 0.03568483141238777;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.03989576403435674;
                        } else {
                          result[0] += 0.1393197019987603;
                        }
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.935600519180298074) ) ) {
                          result[0] += 0.07556307007892421;
                        } else {
                          result[0] += -0.08023467198794745;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.08378248433069524;
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.02998492197118498;
                } else {
                  result[0] += 0.005462984032763316;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.02769110123580263;
        } else {
          result[0] += 0.009263605363625997;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
            result[0] += 0.15415778414305772;
          } else {
            result[0] += -0.01673817909029372;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.01053571701049982) ) ) {
              result[0] += -0.017172870416963228;
            } else {
              result[0] += 0.04236558365976921;
            }
          } else {
            result[0] += -0.15148823440886522;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        result[0] += 0.06284798315212058;
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += 0.04457624130091232;
          } else {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.09537569654550243;
            } else {
              result[0] += 0.009224890600187914;
            }
          }
        } else {
          result[0] += -0.09377230946865264;
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.04384262595826995;
      } else {
        if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.042435407638550693) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
                result[0] += -0.08304936280438341;
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.030769034432029524;
                } else {
                  result[0] += -0.057421288875261094;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.06211993442361622;
              } else {
                result[0] += 0.005475861579351504;
              }
            }
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)28.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.028655311838478022;
              } else {
                result[0] += 0.0276087481911072;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.06080591053278216;
              } else {
                result[0] += 0.007077352013148195;
              }
            }
          }
        } else {
          result[0] += -0.0008756607762287249;
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
        result[0] += -0.0016248957237204394;
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
          result[0] += -0.09142232688596462;
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
              result[0] += 0.15635890638089445;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.05047694275806084;
                } else {
                  result[0] += 0.9127546128623085;
                }
              } else {
                result[0] += 1.4843192282906266;
              }
            }
          } else {
            result[0] += -0.018514615431277828;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.312486410140991655) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.08017798570493001;
          } else {
            result[0] += -0.023846958567756527;
          }
        } else {
          result[0] += -0.02839780234830195;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.01957624399942126;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
              result[0] += -0.07003907436514417;
            } else {
              result[0] += 0.009906057408742713;
            }
          }
        } else {
          result[0] += 0.03332364067291797;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            result[0] += 0.05276139149024024;
          } else {
            result[0] += 0.020024314092846357;
          }
        } else {
          result[0] += -0.044618740103919086;
        }
      } else {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.161602735519410068) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
                result[0] += -0.07464004714228281;
              } else {
                result[0] += 0.178989725327211;
              }
            } else {
              result[0] += 0.0010725272384272675;
            }
          } else {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
              result[0] += -0.022931797016889355;
            } else {
              result[0] += 0.04632459591090534;
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.04291828667479416;
              } else {
                result[0] += -0.08412352950309251;
              }
            } else {
              result[0] += -0.09662136006402416;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += -0.042643940812069556;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.05373600345797813;
              } else {
                result[0] += 0.08393424054006443;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)13.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
            result[0] += 0.06909630767369276;
          } else {
            result[0] += 0.016016841490307614;
          }
        } else {
          result[0] += -0.05473064997578025;
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.02025993498078493;
                } else {
                  result[0] += 0.20656888193801445;
                }
              } else {
                result[0] += -0.10253024172234558;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)20.50000000000000355) ) ) {
                  result[0] += 0.11278644084116585;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                    result[0] += -0.12321423512060894;
                  } else {
                    result[0] += 0.08437398923660155;
                  }
                }
              } else {
                result[0] += -4.371663773339252e-05;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.06097384811372905;
                } else {
                  result[0] += 0.034235244875780654;
                }
              } else {
                result[0] += -0.07955647697582019;
              }
            } else {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                result[0] += -0.09670961249929641;
              } else {
                result[0] += -0.02849757950372433;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
              result[0] += 0.025301955950704804;
            } else {
              result[0] += -0.07107798627865335;
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
              result[0] += 0.12799245635229026;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.13074169265277932;
                  } else {
                    result[0] += -0.0242979317370946;
                  }
                } else {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                      result[0] += -0.018049590280638874;
                    } else {
                      result[0] += -0.16447777383501005;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.11677924784687338;
                      } else {
                        result[0] += 0.003386355852042017;
                      }
                    } else {
                      result[0] += 0.02205307497007101;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.005820190287152032;
                    } else {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                        result[0] += -0.0502167813458799;
                      } else {
                        result[0] += 0.11587688547544127;
                      }
                    }
                  } else {
                    result[0] += -0.06303647425914558;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.11506586372244619;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += -0.05281059007374459;
                      } else {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)12.50000000000000178) ) ) {
                          result[0] += 0.5323180694236649;
                        } else {
                          result[0] += -0.029399528927080518;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += -0.016111029685254182;
                    } else {
                      result[0] += 0.1265931901373453;
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
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.39281225204467951) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.442090511322023261) ) ) {
            result[0] += -0.0019806836001402045;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0093287442155801;
            } else {
              result[0] += -0.146716914734967;
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.011878885209710893;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.005777093378943862;
                } else {
                  result[0] += -0.14811634000720283;
                }
              }
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += 0.07836590147954703;
              } else {
                result[0] += 0.017018467564505126;
              }
            }
          } else {
            result[0] += -0.003691943018759742;
          }
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.0016706794255980516;
                  } else {
                    result[0] += -0.035441197885283694;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                            result[0] += 0.01880409528036858;
                          } else {
                            result[0] += -0.025119699874411545;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                            result[0] += -0.06149496698885618;
                          } else {
                            result[0] += 0.0017361002306033866;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += -0.08100319537362001;
                          } else {
                            result[0] += 0.05230160905551824;
                          }
                        } else {
                          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.10094745984138775;
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                                  result[0] += 0.11452991830415059;
                                } else {
                                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                                    result[0] += -0.21878874911958698;
                                  } else {
                                    result[0] += -0.009312153228913017;
                                  }
                                }
                              } else {
                                result[0] += 0.0440749037061158;
                              }
                            } else {
                              result[0] += -0.09785006276523274;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.0014038234376149174;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.003277486791115155;
                        } else {
                          result[0] += -0.05136694433606488;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0675828373447587;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.08234561052380912;
                  } else {
                    result[0] += -0.003657487131773535;
                  }
                } else {
                  result[0] += -0.0874788652905653;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0013816159310566104;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.05802442792250322;
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.0082141900863475;
                      } else {
                        result[0] += -0.05304733697247295;
                      }
                    }
                  } else {
                    result[0] += 0.03714314983907874;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.01889026179863551;
                    } else {
                      result[0] += -0.07482782248405061;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.0985996074059853;
                    } else {
                      result[0] += -0.025042364562361148;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.01900963531827506;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.01535517446535609;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.04780518270933884;
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                              result[0] += -0.13674432514152024;
                            } else {
                              result[0] += -0.027828897720333423;
                            }
                          } else {
                            result[0] += 0.0053783560553579204;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += 0.005931157768470077;
                      } else {
                        result[0] += 0.050480799561682145;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.043787985990274954;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                result[0] += 0.08085454641417665;
              } else {
                result[0] += -0.01180866307856052;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.09256658691234952;
              } else {
                result[0] += 0.00335434071716142;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += 0.08981206889721369;
                  } else {
                    result[0] += -0.01645971569498179;
                  }
                } else {
                  result[0] += -0.01945293562853102;
                }
              } else {
                result[0] += -0.05759833157876685;
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
              result[0] += 0.0014262601228550685;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += -0.0948787044804309;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.08572874232766348;
                } else {
                  result[0] += -0.09278768217243666;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
        result[0] += -0.005399734651966796;
      } else {
        result[0] += 0.01987697347109342;
      }
    }
  } else {
    result[0] += 0.0068471871335462195;
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.02243120116729179;
        } else {
          result[0] += 0.0999227956911507;
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.849175214767456943) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)29.50000000000000355) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)21.50000000000000355) ) ) {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.005587624098330771;
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.117118597030640537) ) ) {
                            result[0] += -0.012610572386702892;
                          } else {
                            result[0] += 0.11357129141675548;
                          }
                        }
                      } else {
                        result[0] += 0.07844566607675382;
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                        result[0] += -0.16712389522401125;
                      } else {
                        result[0] += 0.07626386647941387;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.012629762919896474;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.07602254315580377;
                      } else {
                        result[0] += -0.023680539420108424;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.004082211681163227;
                    } else {
                      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.1031209647045493;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.05806567000443816;
                            } else {
                              result[0] += 0.031206210902226472;
                            }
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                                result[0] += 0.13088660157704332;
                              } else {
                                result[0] += 0.008713294860730898;
                              }
                            } else {
                              result[0] += 0.026578985198731992;
                            }
                          }
                        } else {
                          result[0] += -0.053384173391059365;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.09472505996301431;
                  }
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)2.891912579536438432) ) ) {
                    result[0] += 0.0006815316900000338;
                  } else {
                    result[0] += 0.10058899285011845;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.004696024090574911;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.015022835756521536;
                    } else {
                      result[0] += -0.05955156539035805;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.004116325673012211;
                } else {
                  result[0] += -0.10035054097975796;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.02371880541356268;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.06806052418719566;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                        result[0] += -0.04070728704830682;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                          result[0] += -0.02513944650959124;
                        } else {
                          result[0] += 0.05433564802794573;
                        }
                      }
                    } else {
                      result[0] += -0.04458522303554247;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0048423276034732995;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                result[0] += -0.043285199452357315;
              } else {
                result[0] += 0.03570780662109418;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.05064961095455717;
              } else {
                result[0] += -0.11703910697057845;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.05448654240064195;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                    result[0] += 0.03978210129849445;
                  } else {
                    result[0] += -0.11182187011765983;
                  }
                } else {
                  result[0] += 0.08514873812205898;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.055454850222925005;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.02576759918351862;
              } else {
                result[0] += -0.05872496191669401;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
          result[0] += -0.013204790238786368;
        } else {
          result[0] += -0.08071512128209747;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += -0.0988828588283246;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.05766536115866142;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += 0.007355559024502568;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
                    result[0] += -0.012722114333523469;
                  } else {
                    result[0] += -0.07790365228666397;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.09883181728140096;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.009608642414677503;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                      result[0] += -0.05084073831449457;
                    } else {
                      result[0] += 0.025816150990447057;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.048607503698049115;
              } else {
                result[0] += 0.04799688229018046;
              }
            } else {
              result[0] += 0.04035249245900255;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              result[0] += -0.038346421728952224;
            } else {
              result[0] += 0.000798832846552537;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
      result[0] += 0.004499908152538393;
    } else {
      result[0] += 0.0379691755220458;
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.04969787597656428) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.015664260094264467;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.18975473972777335;
            } else {
              result[0] += -0.047848144894421574;
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
            result[0] += -0.015447944255405188;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += 0.012704642287666618;
                } else {
                  result[0] += -0.03297527618644486;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.07698901640778627;
                } else {
                  result[0] += 0.059426719385872545;
                }
              }
            } else {
              result[0] += -0.025476622767559604;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += 0.011227121236507476;
            } else {
              result[0] += -0.04423359724350273;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.010111282308892815;
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += 0.0013062122130746696;
                  } else {
                    result[0] += -0.09134335061117753;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                                result[0] += 0.09830857745778715;
                              } else {
                                result[0] += 0.0002481941138718853;
                              }
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                                result[0] += -0.18950780369130352;
                              } else {
                                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                                  result[0] += -0.0470585065999054;
                                } else {
                                  result[0] += 0.06934989689179548;
                                }
                              }
                            }
                          } else {
                            result[0] += -0.0828109373332449;
                          }
                        } else {
                          result[0] += 0.0597242959723882;
                        }
                      } else {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += 0.062127383786372836;
                        } else {
                          result[0] += -0.040567269952769776;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += -0.15416240366571102;
                      } else {
                        result[0] += -0.007907974086253709;
                      }
                    }
                  } else {
                    result[0] += -0.03217280036675665;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.004825329191335462;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.05227878186205603;
                  } else {
                    result[0] += 0.015373129923311732;
                  }
                } else {
                  result[0] += -0.09370083170843534;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += -0.11009746855794492;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.038618429192909226;
            } else {
              result[0] += 0.11835123268468357;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.615554332733154741) ) ) {
            result[0] += 0.03898273045843429;
          } else {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.11374844252330976;
                  } else {
                    result[0] += 0.01068999567710014;
                  }
                } else {
                  result[0] += -0.07044162873593478;
                }
              } else {
                result[0] += 0.029474887082211382;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += 0.010080299733294834;
                  } else {
                    result[0] += -0.077615737522688;
                  }
                } else {
                  result[0] += 0.080857100633275;
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
                    result[0] += -0.0812148781224209;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.039219381014015016;
                    } else {
                      result[0] += 0.06712316266557215;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.456172943115236151) ) ) {
                    result[0] += 0.06292897355924257;
                  } else {
                    result[0] += -0.06984509950778561;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.0011480101041853221;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.0780656092997724;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.025739694636377014;
            } else {
              result[0] += -0.04967176027761121;
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.08781031283330146;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.1563870639220345;
              } else {
                result[0] += 0.009154847595541579;
              }
            }
          } else {
            result[0] += -0.045959457822712955;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
              result[0] += 0.023879284895405656;
            } else {
              result[0] += -0.02562637201631165;
            }
          } else {
            result[0] += 0.033335849496286374;
          }
        } else {
          result[0] += -0.07952098609449201;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.12380192792435266;
        } else {
          result[0] += 0.003760206851579936;
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += -0.0005038943543774825;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          result[0] += -0.049678228986261584;
        } else {
          result[0] += 0.018652626224668407;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.026575016013176484;
          } else {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)11.50000000000000178) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)7.500000000000000888) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += 0.0017536103346196435;
                  } else {
                    result[0] += -0.0944306358339978;
                  }
                } else {
                  result[0] += 0.05301001013480995;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                  result[0] += -0.14205407885225504;
                } else {
                  result[0] += 0.01788529396604641;
                }
              }
            } else {
              result[0] += 0.043262023264150705;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)13.50000000000000178) ) ) {
              result[0] += 0.007456277640568061;
            } else {
              result[0] += -0.04358133789190968;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.027202715691429694;
              } else {
                result[0] += -0.1026193946822073;
              }
            } else {
              result[0] += -0.00887049042263994;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
          result[0] += -0.10743159109323148;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
            result[0] += 0.12611927301495435;
          } else {
            result[0] += 0.04121572871913479;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
        result[0] += -0.08498167409811086;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.06046027670159684;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.627130746841431552) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.022462913950230215;
                } else {
                  result[0] += 0.02985136856296812;
                }
              } else {
                result[0] += 0.09877409998414469;
              }
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.003564894569936503;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
                      result[0] += -0.09211073397827899;
                    } else {
                      result[0] += 0.14557136065976917;
                    }
                  } else {
                    result[0] += 0.05035033436285257;
                  }
                }
              } else {
                result[0] += -0.09723097400211578;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.339013814926148349) ) ) {
            result[0] += -0.0017704667518821278;
          } else {
            result[0] += -0.03475155909030045;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += 0.014436535450245341;
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.007288205811550067;
                } else {
                  result[0] += -0.05749672305617948;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                    result[0] += -0.0552789534351124;
                  } else {
                    result[0] += 0.0943867572673849;
                  }
                } else {
                  result[0] += -0.04153815756830353;
                }
              } else {
                result[0] += 0.03558000326048554;
              }
            }
          } else {
            result[0] += 0.05989116980617011;
          }
        } else {
          result[0] += -0.04333265945913678;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.827801465988160068) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.07614466132411739;
            } else {
              result[0] += 0.0024787118583715433;
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.01302483336885759;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
                if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)8.500000000000001776) ) ) {
                  result[0] += 0.04083744501223192;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.665476083755494052) ) ) {
                    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.019840126336027955;
                      } else {
                        result[0] += 0.04868118428857755;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.339395284652710849) ) ) {
                          result[0] += -0.1834957148776703;
                        } else {
                          result[0] += -0.06769114900082607;
                        }
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
                            result[0] += -0.16677757450562974;
                          } else {
                            result[0] += -0.036730612883154365;
                          }
                        } else {
                          result[0] += 0.016104456765332998;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04588986813817164;
                  }
                }
              } else {
                result[0] += 0.025408913262623345;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.09433873416383846;
          } else {
            result[0] += 0.017278994114250667;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
        result[0] += -0.07088280734828088;
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                result[0] += -0.008923864529139144;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
                  result[0] += 0.028275662636018245;
                } else {
                  result[0] += 0.06273565456507703;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                  result[0] += 0.05870187062801237;
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
                    result[0] += -0.09475800860981648;
                  } else {
                    result[0] += 0.02473343918005269;
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.06432305862289169;
                } else {
                  result[0] += 0.001540380938872174;
                }
              }
            }
          } else {
            result[0] += -0.04113453043470833;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
            result[0] += -0.06297964549467062;
          } else {
            result[0] += -0.002721966912121728;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.0003481282594056715;
    } else {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
            result[0] += 0.16650741302831326;
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.16211979445769675;
            } else {
              result[0] += -0.005995366107208463;
            }
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.04702424219281891;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.01302411862024662;
              } else {
                result[0] += -0.09668057447801186;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.74696540832519709) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.037822514944838355;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.047098342397960805;
                    } else {
                      result[0] += 0.01829881565712078;
                    }
                  } else {
                    result[0] += 0.03590562330190202;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.11575897656195938;
                } else {
                  result[0] += -0.01413180533296356;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.007931466001928214;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.14801472183859485;
                } else {
                  result[0] += 0.12561842775710572;
                }
              } else {
                result[0] += 0.06043788854886866;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.04044935711691178;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += -0.07944116071830737;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.0352981635038566;
                  } else {
                    result[0] += 0.08738720966181476;
                  }
                }
              } else {
                result[0] += -0.0809051440501996;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.007083275401316769;
              } else {
                result[0] += -0.03300385153937054;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.07399487495078738;
                    } else {
                      result[0] += 0.010858449818756005;
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.10406415362216917;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.58381557464599787) ) ) {
                        result[0] += -0.06792880400114636;
                      } else {
                        result[0] += 0.03672847986994122;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.061251803255162045;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.02011217039571422;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.0003830570421642441;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.589026927947998269) ) ) {
                          result[0] += 0.0612453280933031;
                        } else {
                          result[0] += -0.018046243000514207;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.030870736755309575;
                  } else {
                    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += 0.010734763835963612;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += -0.11595400034208624;
                        } else {
                          result[0] += -0.027701698101910446;
                        }
                      }
                    } else {
                      result[0] += 0.005035519035073481;
                    }
                  }
                } else {
                  result[0] += -0.05918879288187784;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                    result[0] += 0.13133355378631223;
                  } else {
                    result[0] += 0.04014350838528206;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.08999514326370522;
                    } else {
                      result[0] += -0.047028417366796256;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                        result[0] += 0.06290867351602603;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                          result[0] += -0.06713470290715474;
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                              result[0] += -0.047961854602028696;
                            } else {
                              result[0] += 0.13977038349420268;
                            }
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                              result[0] += 0.01982865961731003;
                            } else {
                              result[0] += -0.18754673561029955;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                        result[0] += -0.12614297361611385;
                      } else {
                        result[0] += 0.05875628743289218;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.015941205981979813;
                  } else {
                    result[0] += -0.07196238436783402;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                          result[0] += -0.04646126811829369;
                        } else {
                          result[0] += 0.02005027910859791;
                        }
                      } else {
                        result[0] += -0.039148069577059325;
                      }
                    } else {
                      result[0] += -0.10520959216539805;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.09300264845241091;
                    } else {
                      result[0] += 0.008176215072574376;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.09864668028660796;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.04036986263637579;
    } else {
      result[0] += 0.007896054072800088;
    }
  }
}

