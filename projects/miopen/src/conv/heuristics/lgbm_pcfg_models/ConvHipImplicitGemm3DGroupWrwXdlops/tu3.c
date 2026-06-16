
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += 0.06618023343811634;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += 0.0010794701411009167;
            } else {
              result[0] += 0.02312832040172585;
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.094205617904663974) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.0769021223234931;
                    } else {
                      result[0] += -0.005517923380254719;
                    }
                  } else {
                    result[0] += 0.03318603755477829;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.011472531907178883;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += -0.01481106284033324;
                      } else {
                        result[0] += -0.05219181890848956;
                      }
                    }
                  } else {
                    result[0] += -0.05639209796309738;
                  }
                }
              } else {
                result[0] += -0.045963262702591975;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.09584710455882162;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.023828571112710587;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += -0.06390334056299263;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += 0.039482369044412546;
                        } else {
                          result[0] += -0.09994609798575439;
                        }
                      } else {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                            result[0] += -0.04432963914720266;
                          } else {
                            result[0] += -0.006465610060344232;
                          }
                        } else {
                          result[0] += 0.006502035119132885;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.002341830226867284;
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.004389698082016563;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.05022771653484157;
                  } else {
                    result[0] += 0.04263157901839108;
                  }
                } else {
                  result[0] += -0.04958482395463678;
                }
              } else {
                result[0] += -0.042257654840969716;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
            result[0] += 0.04258447790151703;
          } else {
            result[0] += -0.09062457695007842;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
              result[0] += -0.0646708303057085;
            } else {
              result[0] += 0.12823370044780405;
            }
          } else {
            result[0] += -0.07742713334831608;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            result[0] += -0.10745200673928007;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.09303667691307047;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                result[0] += 0.033700066725257494;
              } else {
                result[0] += -0.025537842050456696;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.05076844715663829;
          } else {
            result[0] += 0.013296018732457976;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
          result[0] += 0.06388603773958768;
        } else {
          result[0] += -0.037224694482042464;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
      result[0] += -0.02564461072570308;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.06050853686433835;
            } else {
              result[0] += 0.05717494816686384;
            }
          } else {
            result[0] += -0.07060837290510143;
          }
        } else {
          result[0] += -0.05464309808564663;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.572941064834595615) ) ) {
              result[0] += -0.09055290396257856;
            } else {
              result[0] += 0.09827491558318585;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
              result[0] += -0.07691455784522336;
            } else {
              result[0] += 0.17045760274157018;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.14029987902611557;
              } else {
                result[0] += 0.04705037389735579;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                  result[0] += -0.04897229623006087;
                } else {
                  result[0] += 0.028807564809554098;
                }
              } else {
                result[0] += 0.03962295672250241;
              }
            }
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
              result[0] += -0.0022078911418883657;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.05480805021754501;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.05946673101836567;
                    } else {
                      result[0] += 0.03297664499637761;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                    result[0] += 0.05104017745098069;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += 0.48534135839992854;
                      } else {
                        result[0] += -0.007056136933914742;
                      }
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.9442821413058056;
                        } else {
                          result[0] += 2.3912705134063503;
                        }
                      } else {
                        result[0] += 0.17001654927621182;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0026819576030652345;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
          result[0] += 0.0021809598864636686;
        } else {
          result[0] += 0.02446939816158372;
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.790835380554201) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.011766622468631891;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.009702263916914408;
                } else {
                  result[0] += -0.08071794999729279;
                }
              }
            } else {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.011617873801480834;
                  } else {
                    result[0] += 0.049983646768289314;
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                          result[0] += -0.016867211140929053;
                        } else {
                          result[0] += 0.026397569606385864;
                        }
                      } else {
                        result[0] += -0.14537067560249217;
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
                          result[0] += -0.0068013948050602655;
                        } else {
                          result[0] += -0.08885142859886577;
                        }
                      } else {
                        result[0] += -0.056489060357043125;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.12619976866685434;
                    } else {
                      result[0] += 0.08067718237515398;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.060653967616782024;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += 0.027903049400358895;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.029077566594654422;
                        } else {
                          result[0] += 0.012259552678518887;
                        }
                      } else {
                        result[0] += -0.055267189807947516;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04192417078853959;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.0200465572077213;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                                result[0] += -0.06623175274857017;
                              } else {
                                result[0] += 0.08662608981792691;
                              }
                            } else {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.947818994522095615) ) ) {
                                result[0] += 0.022572500600701213;
                              } else {
                                result[0] += -0.1438668767024173;
                              }
                            }
                          } else {
                            result[0] += -0.03705516008411226;
                          }
                        } else {
                          result[0] += -0.003148083234660893;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.05071630541189978;
                      } else {
                        result[0] += 0.015740390875881565;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                  result[0] += -0.11770109954283206;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.035335057770158516;
                  } else {
                    result[0] += -0.07700392350823507;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.671854496002199042) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.011623616093180167;
                  } else {
                    result[0] += -0.06939643132382965;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.024665729729431;
                  } else {
                    result[0] += -0.055184161661673295;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.07755757754722063;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                    result[0] += -0.055761990554708235;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.05009834444407468;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.05203607541420018;
                        } else {
                          result[0] += -0.019658298808569383;
                        }
                      }
                    } else {
                      result[0] += 0.017221092974991328;
                    }
                  }
                } else {
                  result[0] += -0.08274950390023221;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -0.003581922871653068;
          } else {
            result[0] += -0.06402735901226718;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.075411182657529;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.022650143192509326;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.06609186811114316;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
                result[0] += 0.12138846492075384;
              } else {
                result[0] += -0.007176685322647662;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
          result[0] += 0.058676472431022454;
        } else {
          result[0] += -0.025096813903421403;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
      result[0] += -0.013876319830149218;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.0341941877772256;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.947818994522095615) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
              result[0] += 0.05412724123797831;
            } else {
              result[0] += -0.04391337326774587;
            }
          } else {
            result[0] += -0.05744758427407781;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
                result[0] += -0.0891529406134099;
              } else {
                result[0] += 0.08551893061537402;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                result[0] += -0.0885385016616216;
              } else {
                result[0] += 0.14772986839621724;
              }
            }
          } else {
            result[0] += 0.022503077698068613;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += 0.06482902377857457;
        } else {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.02945462879615242;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.004348364792224204;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.05471160303380264;
                    } else {
                      result[0] += -0.1786474731138491;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.015428325956899386;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.41262340545654475) ) ) {
                        result[0] += -0.02226188106093192;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += 0.018485397424464558;
                        } else {
                          result[0] += 0.09748408567241329;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
                result[0] += -0.02485352730104929;
              } else {
                result[0] += 0.030755980889353204;
              }
            }
          } else {
            result[0] += 0.10275288735046342;
          }
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += 0.020556614229444235;
                  } else {
                    result[0] += -0.03643904227915217;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.07649795039088603;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                      result[0] += -0.024482086938803084;
                    } else {
                      result[0] += 0.11470716728563189;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.02029972121893098;
                  } else {
                    result[0] += -0.017825399177915608;
                  }
                } else {
                  result[0] += -0.03205751681617017;
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.06619577073530516;
                } else {
                  result[0] += -0.020860765950115268;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.03897199766911304;
                } else {
                  result[0] += 0.021920416009479877;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0013799261187328715;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.13527198619826808;
                      } else {
                        result[0] += -0.0058794085226300296;
                      }
                    } else {
                      result[0] += -0.029114834393366373;
                    }
                  } else {
                    result[0] += -0.09303689683348834;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.0003437473902469449;
                  } else {
                    result[0] += -0.027124350174104063;
                  }
                }
              }
            } else {
              result[0] += -0.07716803988889956;
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.0774201427242732;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                  result[0] += -0.025682775375033114;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.11872778152681099;
                  } else {
                    result[0] += 0.006322096603064519;
                  }
                }
              } else {
                result[0] += -0.04375142325518728;
              }
            }
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.018903257955774404;
                } else {
                  result[0] += -0.045682288561733;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.07379936439520335;
                  } else {
                    result[0] += 0.006920322763233035;
                  }
                } else {
                  result[0] += 0.026419542616859405;
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.08977701204238595;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += -0.05310727512220386;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                          result[0] += 0.04676574509825193;
                        } else {
                          result[0] += -0.03325487968423416;
                        }
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.017737266746510983;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.043548787189020194;
                          } else {
                            result[0] += 0.08140700854603655;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                        result[0] += -0.033255041419678846;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                            result[0] += -0.03481134739247742;
                          } else {
                            result[0] += 0.009835948281451571;
                          }
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                            result[0] += -0.001339377968122081;
                          } else {
                            result[0] += -0.07090321429363834;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0028652242992731666;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
          result[0] += 0.002991211235165685;
        } else {
          result[0] += 0.10645262921726051;
        }
      } else {
        result[0] += 0.04762569828263492;
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0067436879404252125;
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        result[0] += 0.06205172421384045;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          result[0] += -0.04003212006018691;
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.030477985881991788;
          } else {
            result[0] += -0.00018205739300146238;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
            result[0] += 0.09268532618054412;
          } else {
            result[0] += 0.01651373077511949;
          }
        } else {
          result[0] += 0.0013714726957649193;
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += 0.003252205317017065;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.012263201412450344;
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.11471128107796319;
                          } else {
                            result[0] += -0.024874410795547894;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                              result[0] += -0.022887732182042476;
                            } else {
                              result[0] += 0.18210522679932695;
                            }
                          } else {
                            result[0] += -0.07511208386367436;
                          }
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.028976239485129535;
                          } else {
                            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                              result[0] += -0.004844461630786234;
                            } else {
                              result[0] += 0.06337351739664691;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += 0.01990712596489332;
                    }
                  } else {
                    result[0] += 0.030330629985571452;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += 0.016865399215486285;
                  } else {
                    result[0] += -0.1272537548346274;
                  }
                }
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                        result[0] += 0.005735552933112201;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += 0.12361439822540572;
                        } else {
                          result[0] += 0.018664124160339298;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.006102499232497841;
                      } else {
                        result[0] += -0.07186838325403504;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.08873071019477108;
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.058229314955369385;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                            result[0] += 0.06139242388751533;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.597218394279480425) ) ) {
                              result[0] += -0.009537995108618653;
                            } else {
                              result[0] += -0.059331922663623964;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                            result[0] += -0.0228422519524365;
                          } else {
                            result[0] += 0.017677433569171955;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.10477203980794832;
                    } else {
                      result[0] += -0.045928217054838794;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
                      result[0] += 0.005861255824556857;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.007658262646542584;
                      } else {
                        result[0] += -0.07392620673671114;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)17.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.02616319412271891;
                  } else {
                    result[0] += 0.06899928103581883;
                  }
                } else {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                    result[0] += 0.013442712291601109;
                  } else {
                    result[0] += -0.10162714514075469;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.03852571992331727;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.048242506787026795;
                  } else {
                    result[0] += 0.0013266269911478604;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                result[0] += 0.00940390720039632;
              } else {
                result[0] += -0.06029957974361249;
              }
            }
          }
        } else {
          result[0] += 0.00015605454744455093;
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
            result[0] += 0.008188596278805962;
          } else {
            result[0] += 0.11279865201973911;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.010070383814144927;
          } else {
            result[0] += 0.0715981464292039;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
          result[0] += -0.09278964916165024;
        } else {
          result[0] += -0.005751701899657198;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
          result[0] += -0.0017545940337561116;
        } else {
          result[0] += 0.07112478258188626;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.48738741874694913) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.02707855368169464;
          } else {
            result[0] += 0.03160015157058643;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
              result[0] += -0.11361574980337125;
            } else {
              result[0] += -0.009638578045500059;
            }
          } else {
            result[0] += -0.10569291018928273;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.13404753615816087;
        } else {
          result[0] += 0.04630196503317105;
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
            result[0] += 0.003525859551786598;
          } else {
            result[0] += -0.0841932685923888;
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.03963413813640868;
          } else {
            result[0] += 0.012317634394171321;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.2687106132507342) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.004125490281347653;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.14698185727710353;
              } else {
                result[0] += -0.0738828758162233;
              }
            }
          } else {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.012484818080487898;
            } else {
              result[0] += 0.07022200053321308;
            }
          }
        } else {
          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.02350066059637626;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.10767721603644108;
              } else {
                result[0] += -0.048521183222504684;
              }
            }
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.07079019330915547;
            } else {
              result[0] += -0.01866609821143973;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                  result[0] += 0.1331068715101687;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += -0.08116183356300154;
                  } else {
                    result[0] += 0.20132909966017665;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += 0.06191916577328643;
                  } else {
                    result[0] += -0.04318779038630021;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.06803939573602781;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.09116103125597441;
                    } else {
                      result[0] += 0.0662920594171719;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.05496257675915955;
            }
          } else {
            result[0] += -0.06432411807330209;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                      result[0] += 0.014619263692507795;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                          result[0] += -0.044872845515584035;
                        } else {
                          result[0] += 0.054473439123501805;
                        }
                      } else {
                        result[0] += -0.009325672941378356;
                      }
                    }
                  } else {
                    result[0] += 0.009617202700474937;
                  }
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.03672033196939297;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                      result[0] += 0.011002979217352448;
                    } else {
                      result[0] += -0.023989431371458692;
                    }
                  }
                }
              } else {
                result[0] += -0.05535981284440694;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)88.00000000000001421) ) ) {
                      result[0] += 0.026722029774796;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.039720773696899636) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.07934941491291053;
                          } else {
                            result[0] += -0.06653459537547467;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.842307567596437323) ) ) {
                            result[0] += -0.10727734329019428;
                          } else {
                            result[0] += 0.0026027501775087543;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.11595218175430251;
                        } else {
                          result[0] += -0.012841592055080807;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                      result[0] += -0.03990952882801702;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += -0.056547389831446096;
                      } else {
                        result[0] += 0.0516192677079666;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += -0.02961771447730187;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += 0.05427804290581654;
                      } else {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                            result[0] += 0.03244694938579285;
                          } else {
                            result[0] += -0.051638521335885125;
                          }
                        } else {
                          result[0] += 0.02209765162204758;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                        result[0] += -0.049742060202818995;
                      } else {
                        result[0] += 0.047737439987011215;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.19876670837402521) ) ) {
                  result[0] += -0.06011110565726137;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0058806443888806004;
                  } else {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += 0.031013196853311737;
                      } else {
                        result[0] += -0.04052918838587499;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.1264837836182642;
                      } else {
                        result[0] += -0.010286445877304444;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.05360144472986627;
                } else {
                  result[0] += 0.02960236240689021;
                }
              } else {
                result[0] += -0.06556980043341086;
              }
            } else {
              if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                  result[0] += 0.027291965121232345;
                } else {
                  result[0] += -0.03912357468893313;
                }
              } else {
                result[0] += 0.07973286664102146;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += 0.004667782619205738;
      } else {
        result[0] += 0.04483338722018793;
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
      result[0] += -0.015502378762944394;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        result[0] += -0.032006384302590014;
      } else {
        result[0] += 0.02288019054785239;
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0644651355181507;
            } else {
              result[0] += 0.0015424454169928516;
            }
          } else {
            result[0] += -0.02421036512378271;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.022474259022466903;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.08932048455912886;
            } else {
              result[0] += 0.011830808127363458;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += 0.0011379762344094974;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.79870843887329279) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.030331672864900645;
                      } else {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.001019252768936501;
                        } else {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                            result[0] += -0.05668599256678322;
                          } else {
                            result[0] += 0.020361126943479455;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                          result[0] += 0.047982415836455516;
                        } else {
                          result[0] += -0.053017897338547115;
                        }
                      } else {
                        result[0] += -0.08174195354092226;
                      }
                    }
                  } else {
                    result[0] += -0.052381448794883136;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.834949493408204901) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                      result[0] += 0.0054889506626027255;
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.21184828730959607;
                      } else {
                        result[0] += 0.01161321773122662;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.65906000137329146) ) ) {
                        result[0] += -0.08250524407328964;
                      } else {
                        result[0] += 0.1859116093669415;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.059960903130445276;
                          } else {
                            result[0] += -0.027896852780108186;
                          }
                        } else {
                          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.030938320392279724;
                          } else {
                            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                                result[0] += -0.030586337149367523;
                              } else {
                                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += -0.0477871326434678;
                                } else {
                                  result[0] += 0.06737001691570224;
                                }
                              }
                            } else {
                              result[0] += -0.060513510013310734;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.03263303729076637;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                        result[0] += 0.008279321548653568;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += 0.013584931020065983;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                            result[0] += 0.011902663560351343;
                          } else {
                            result[0] += -0.0814095477107977;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                        result[0] += -0.03177355436031832;
                      } else {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.07310972362187333;
                        } else {
                          result[0] += 0.07095334482731905;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.025635846328192765;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.06519283707949407;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.029191594703696325;
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.047963310808815274;
                      } else {
                        result[0] += -0.0606708493846717;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.11517142815469034;
              } else {
                result[0] += -0.004602904576856388;
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.028197954976657592;
                } else {
                  result[0] += -0.07018124790598582;
                }
              } else {
                result[0] += 0.009526714364902406;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
            result[0] += -0.03784414128576734;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                  result[0] += -0.058836692032750965;
                } else {
                  result[0] += 0.049616454402562;
                }
              } else {
                result[0] += 0.0710050637879571;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.019955073749440486;
              } else {
                result[0] += -0.1427202458362322;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
          result[0] += 0.019659267769452482;
        } else {
          result[0] += -0.0948548704831119;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.009520895956241018;
        } else {
          result[0] += 0.050413143373997396;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        result[0] += 0.028254836314723548;
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.02527721826290845;
          } else {
            result[0] += 0.024997806976234438;
          }
        } else {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += -0.006581673987982673;
          } else {
            result[0] += -0.08143984887725328;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
          result[0] += -0.000752288410933208;
        } else {
          result[0] += 0.03434642024899919;
        }
      } else {
        result[0] += -0.0005372312968640583;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.05886165486390624;
            } else {
              result[0] += 0.001158164920130139;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += 0.001340003243297667;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.09532203887976406;
              } else {
                result[0] += 0.001756846314482365;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.022110139099095403;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += -0.08554009407252806;
            } else {
              result[0] += 0.011608521603546952;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.008512116808130915;
                  } else {
                    result[0] += -0.0004026219190798505;
                  }
                } else {
                  result[0] += -0.052364802184585016;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.849175214767456943) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.0051636690726215555;
                      } else {
                        result[0] += 0.06454326983367002;
                      }
                    } else {
                      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.07316150879561792;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += 0.0016228155832679744;
                        } else {
                          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
                              result[0] += -0.07164980181877487;
                            } else {
                              result[0] += 0.054214398993012475;
                            }
                          } else {
                            result[0] += -0.09260712100477692;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.1387999984621142;
                          } else {
                            result[0] += 0.5820507938274154;
                          }
                        } else {
                          result[0] += 0.06685737972113179;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.06491028353889801;
                        } else {
                          result[0] += -0.017672530819965445;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                        result[0] += -0.062235863979229804;
                      } else {
                        result[0] += 0.02763914630030373;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)10.50000000000000178) ) ) {
                    result[0] += -0.05464925579313451;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += 0.05566042033103499;
                    } else {
                      result[0] += -0.04759017798042788;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.015110864511213488;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                        result[0] += 0.05290547579818475;
                      } else {
                        result[0] += -0.024795026392280397;
                      }
                    } else {
                      result[0] += 0.004284610899754934;
                    }
                  }
                } else {
                  result[0] += -0.0182794594991464;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.08928520162508562;
                  } else {
                    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += -0.044252175202850026;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                        if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)30.50000000000000355) ) ) {
                          result[0] += -0.041194715908459405;
                        } else {
                          result[0] += 0.15799014932185523;
                        }
                      } else {
                        result[0] += -0.10525596894081508;
                      }
                    }
                  }
                } else {
                  result[0] += -0.08751980863568179;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.141444921493531162) ) ) {
                  result[0] += -0.1596981316750118;
                } else {
                  result[0] += 0.01612070265606689;
                }
              } else {
                result[0] += 0.03594924697217049;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.947818994522095615) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.07927711270125933;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.161602735519410068) ) ) {
                    result[0] += 0.059324096405784954;
                  } else {
                    result[0] += -0.08127199373176129;
                  }
                }
              } else {
                result[0] += -0.15335536127035618;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += 0.0008504538999474225;
          } else {
            result[0] += -0.06153990834232967;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
          result[0] += 0.001100908560740604;
        } else {
          result[0] += 0.09187486649377472;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.012325274277509693;
          } else {
            result[0] += 0.06528880558368695;
          }
        } else {
          result[0] += -0.01921530625773106;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += -0.014127732919623832;
          } else {
            result[0] += -0.1275306317427398;
          }
        } else {
          result[0] += 0.008098462823325835;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
            result[0] += 0.059893675884179626;
          } else {
            result[0] += -0.08944329282209784;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.063810348510743076) ) ) {
              result[0] += -0.07947262871096823;
            } else {
              result[0] += 0.002671940678672348;
            }
          } else {
            result[0] += 0.018462667741858483;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        result[0] += 0.036726642241167624;
      } else {
        result[0] += 0.01315930913920149;
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.394753932952881748) ) ) {
            result[0] += 0.015779229107286314;
          } else {
            result[0] += 0.1125241641990038;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += 0.000899229989199822;
            } else {
              result[0] += -0.03073058929621425;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.017713694587388525;
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                result[0] += -0.08329882893053858;
              } else {
                result[0] += 0.01003525637700752;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
            result[0] += -0.0033948645267998507;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
              result[0] += -0.07508604431242587;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                    result[0] += -0.048524307457029625;
                  } else {
                    result[0] += 0.006152387373982545;
                  }
                } else {
                  result[0] += -0.05253322692178819;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.06741342885514041;
                } else {
                  result[0] += -0.057613575588859704;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.283100605010987216) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.005248546671682206;
                    } else {
                      result[0] += 0.1664000641869019;
                    }
                  } else {
                    result[0] += 0.00029881879127573825;
                  }
                } else {
                  result[0] += 0.054825304860889906;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
                  result[0] += -0.008773157215577627;
                } else {
                  result[0] += -0.05812213238961905;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                result[0] += 0.006270554031651553;
              } else {
                result[0] += -0.07699512455807224;
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.03503762364891838;
              } else {
                result[0] += 0.09428142179550653;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.032388780097204875;
                    } else {
                      result[0] += 0.07050414489008017;
                    }
                  } else {
                    result[0] += 0.016101999762029123;
                  }
                } else {
                  result[0] += -0.07282373788490641;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.0003827015468929125;
                  } else {
                    result[0] += -0.10287015537831037;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                    result[0] += -0.07853981377615098;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.020303823596740836;
                    } else {
                      result[0] += 0.01811522062520084;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
              result[0] += -0.04835871641143066;
            } else {
              result[0] += 0.07008476207922397;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += 0.0210072082778884;
            } else {
              result[0] += -0.09824394860003627;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                result[0] += -0.060229102304440424;
              } else {
                result[0] += 0.06413647297119258;
              }
            } else {
              result[0] += -0.040369257132753016;
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.06852830114935768;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.0009079128010748101;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.08097520024662103;
                } else {
                  result[0] += 0.0679389733309948;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          result[0] += 0.013620522207972259;
        } else {
          result[0] += -0.05631206628103175;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
          result[0] += -0.009583614823218988;
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0032225176596110414;
          } else {
            result[0] += 0.08902231763847514;
          }
        }
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)12.50000000000000178) ) ) {
              result[0] += 0.025798756326619134;
            } else {
              result[0] += -0.0010051916934433493;
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += 0.07719825312840872;
            } else {
              result[0] += -0.09249548035434674;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += 0.0031781726318887067;
          } else {
            result[0] += -0.053281395075766264;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          result[0] += 0.04916590393930876;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.07665549934340737;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                  result[0] += -0.05356292827629915;
                } else {
                  result[0] += 0.09951394652790745;
                }
              }
            } else {
              result[0] += -0.03806575506868627;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.036049604415894443) ) ) {
              result[0] += 0.0195694660489118;
            } else {
              result[0] += 0.08694715230162596;
            }
          }
        }
      } else {
        result[0] += 5.779606031175856e-05;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
        result[0] += 0.0009176977910616901;
      } else {
        result[0] += 0.09282494971520151;
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.06205230209993543;
                        } else {
                          result[0] += 0.024427060123866848;
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += 0.036436715916741316;
                          } else {
                            result[0] += -0.07807530550927055;
                          }
                        } else {
                          result[0] += 0.06025883951339598;
                        }
                      }
                    } else {
                      result[0] += -0.11835546780227091;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.010451862331994147;
                    } else {
                      result[0] += 0.05660962530538314;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.06137574998392997;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                      result[0] += 0.03667045380210101;
                    } else {
                      result[0] += -0.038208819318011907;
                    }
                  }
                }
              } else {
                result[0] += -0.004569189828279973;
              }
            } else {
              result[0] += -0.12801829491863814;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                      result[0] += -0.008148553237931653;
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.935600519180298074) ) ) {
                          result[0] += 0.011194993169152312;
                        } else {
                          result[0] += -0.08304976293275294;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                          if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.10254161723806761;
                          } else {
                            result[0] += 0.012003722605530181;
                          }
                        } else {
                          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.08614930960138326;
                          } else {
                            result[0] += -0.05662414583066955;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.06358638140072707;
                  }
                } else {
                  result[0] += 0.0485231719293118;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.03539650079120881;
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.019094614916172384;
                      } else {
                        result[0] += 0.13404880340803002;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.08936643527633935;
                    } else {
                      result[0] += -0.041210891365015545;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
                    result[0] += 0.015318722534367783;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.04827702413141573;
                        } else {
                          result[0] += -0.00815232327599037;
                        }
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                          result[0] += 0.02327045773977579;
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.10472046871582741;
                          } else {
                            result[0] += -0.0021937498630484062;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.013266363230303613;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.6749763488769549) ) ) {
                          result[0] += 0.04967079731657728;
                        } else {
                          result[0] += -0.037897077778897406;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.05019153930786;
                  } else {
                    if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)7.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                          result[0] += 0.07729115075659519;
                        } else {
                          result[0] += -0.07379692544602955;
                        }
                      } else {
                        result[0] += 0.08035682287040519;
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.07636665002996632;
                      } else {
                        result[0] += 0.03705430855021783;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.08879954316475108;
                  } else {
                    result[0] += -0.04041251832832938;
                  }
                }
              } else {
                result[0] += 0.04942894413606305;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.15233457701676678;
                } else {
                  result[0] += 0.06492280474388375;
                }
              } else {
                result[0] += 0.01686912871031949;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.013286441821047504;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.009145671896926421;
                } else {
                  result[0] += 0.0627881829952987;
                }
              }
            }
          } else {
            result[0] += -0.02396668423545571;
          }
        }
      } else {
        result[0] += -0.09412444500794903;
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
      result[0] += -0.01364117812754885;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
        result[0] += -0.09757660861288545;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.040638412675389916;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += -0.07581340226722962;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.08340026480738487;
                } else {
                  result[0] += 0.001765942933714147;
                }
              } else {
                result[0] += 0.08005982311559501;
              }
            }
          } else {
            result[0] += 0.02254208898606851;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[46].missing != -1) && (data[46].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.01581353919337272;
            } else {
              result[0] += 0.05908205984126205;
            }
          } else {
            result[0] += -0.0667541498425872;
          }
        } else {
          result[0] += 0.0007959241029276498;
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
            result[0] += -0.008341284607274369;
          } else {
            result[0] += -2.9907108247719108e-06;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
              result[0] += -0.013054544220474274;
            } else {
              result[0] += -0.07881780455114262;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.283100605010987216) ) ) {
                    result[0] += 0.07490441810346456;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.011601979295452619;
                    } else {
                      result[0] += 0.0781757387134936;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.015709295866509064;
                  } else {
                    result[0] += 0.030203597454505068;
                  }
                }
              } else {
                result[0] += -0.06810113365744398;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += 0.0006202631886180713;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.07706828784977743;
                  } else {
                    result[0] += 0.1033407888267771;
                  }
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.03644659586513747;
                  } else {
                    result[0] += -0.13430986996705352;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.042435407638550693) ) ) {
              result[0] += -0.06535598997893464;
            } else {
              result[0] += 0.05965350538599619;
            }
          } else {
            result[0] += -0.04915862855253348;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += -0.0451299761242944;
              } else {
                result[0] += 0.07633004262153162;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.018841846988186303;
              } else {
                result[0] += -0.07155194256681359;
              }
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.07105000186786346;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.005760289437239468;
              } else {
                result[0] += 0.060715974554064384;
              }
            }
          }
        }
      } else {
        result[0] += -0.029278446113388736;
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        result[0] += 0.032869382151470394;
      } else {
        result[0] += -0.0525678531963263;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
            result[0] += 0.07850513233478007;
          } else {
            result[0] += -0.08055973945165816;
          }
        } else {
          result[0] += -0.02122949168074102;
        }
      } else {
        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
              result[0] += 0.044939294965807244;
            } else {
              result[0] += -0.07309380142717252;
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.12616154951793457;
                } else {
                  result[0] += -0.009649393185262518;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.09063778318021781;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.03711230181925056;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
                        result[0] += -0.03762511334142902;
                      } else {
                        result[0] += 0.047844816090118814;
                      }
                    }
                  }
                } else {
                  result[0] += 0.08355866229164338;
                }
              }
            } else {
              result[0] += 0.02772387971195267;
            }
          }
        } else {
          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)31.50000000000000355) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.912734985351563388) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
                  result[0] += -0.087652394200784;
                } else {
                  result[0] += -0.01861088226114132;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.03562535992833254;
                    } else {
                      result[0] += 0.09965512454408489;
                    }
                  } else {
                    result[0] += -0.014499175194246459;
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)11.50000000000000178) ) ) {
                      result[0] += 0.0793275858674213;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += 0.042167419314907764;
                      } else {
                        result[0] += -0.07890939303336651;
                      }
                    }
                  } else {
                    result[0] += 0.03419557093366625;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                result[0] += 0.16018448181741357;
              } else {
                result[0] += -0.06771523014855384;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                  result[0] += -0.14288640137822875;
                } else {
                  result[0] += 0.026838767047558526;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.08520515568652842;
                } else {
                  result[0] += -0.066153505278687;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.0684506591091473;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.161602735519410068) ) ) {
                  result[0] += 0.05610038351087983;
                } else {
                  result[0] += -0.06215785497012352;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)15.50000000000000178) ) ) {
      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)14.50000000000000178) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0012386946224157667;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)7.500000000000000888) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)9.500000000000001776) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)8.500000000000001776) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.001347437606841511;
                        } else {
                          result[0] += 0.05701094328911423;
                        }
                      } else {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.031546890279689835;
                          } else {
                            result[0] += 0.01376084580576465;
                          }
                        } else {
                          if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                            result[0] += -0.0062565935964709165;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                              result[0] += -0.013581636580562979;
                            } else {
                              result[0] += -0.08175470148922173;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.10970334965569778;
                      } else {
                        result[0] += 0.009699455515690028;
                      }
                    }
                  } else {
                    result[0] += 0.023759644131531525;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
                        result[0] += -0.0167417503389526;
                      } else {
                        result[0] += 0.01648211130875384;
                      }
                    } else {
                      result[0] += 0.005081670239827255;
                    }
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.04614931804809682;
                    } else {
                      result[0] += 0.013171608777262213;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.01848956239085292;
                } else {
                  result[0] += -0.021119930299848197;
                }
              }
            } else {
              result[0] += -0.09322970815731306;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
            result[0] += -0.036299612756561836;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                  result[0] += -0.05750689455712424;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04655761308896495;
                  } else {
                    result[0] += 0.09700971952739638;
                  }
                }
              } else {
                result[0] += 0.06526076289815085;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.015587254483484645;
              } else {
                result[0] += -0.13335080718274384;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.019804783857423278;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.053840725939487605;
              } else {
                result[0] += -0.0642583720077786;
              }
            } else {
              result[0] += 0.0705838403219822;
            }
          }
        } else {
          result[0] += -0.04971526879783403;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
        if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)8.500000000000001776) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.03942623199955528;
          } else {
            result[0] += 0.08217260957357352;
          }
        } else {
          result[0] += 0.055103237165725176;
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                result[0] += -0.03567214735008023;
              } else {
                result[0] += 0.03194198514072334;
              }
            } else {
              result[0] += -0.0979450150627435;
            }
          } else {
            result[0] += -0.07247143925877474;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
            result[0] += 0.0017731528249136281;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)8.500000000000001776) ) ) {
                result[0] += -0.08116559279802106;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0179038800964272;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += 0.12004396309384487;
                  } else {
                    result[0] += -0.04469635836062741;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += -0.06912679605650013;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.8442629078382752;
                  } else {
                    result[0] += 0.12364550764756853;
                  }
                }
              } else {
                result[0] += -0.09176755977243062;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.827801465988160068) ) ) {
          result[0] += 0.02030908070466965;
        } else {
          result[0] += 0.08254672581346914;
        }
      } else {
        result[0] += -0.02331714356263638;
      }
    } else {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)12.50000000000000178) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += 0.031008853046843324;
          } else {
            result[0] += -0.054566892615185505;
          }
        } else {
          result[0] += -0.007922438799503042;
        }
      } else {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.04402474153104394;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
              result[0] += -0.037923824657593384;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.09424485085965988;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                      result[0] += -0.03577008997668027;
                    } else {
                      result[0] += 0.09204824832031941;
                    }
                  } else {
                    result[0] += 0.05232584947814187;
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.05856483486276971;
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)16.50000000000000355) ) ) {
                    result[0] += 0.03524146824626333;
                  } else {
                    result[0] += -0.10484188562166719;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -1.458851320419799e-05;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)7.000000000000000888) ) ) {
        result[0] += -0.00039042460061547906;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.087193608283997026) ) ) {
              result[0] += -0.05201396934792357;
            } else {
              result[0] += 0.17061093030860627;
            }
          } else {
            result[0] += -0.010790047350834414;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              result[0] += -0.05997642955467433;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                result[0] += -0.09622311830852738;
              } else {
                result[0] += 0.06438955727293504;
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0012106786254509356;
            } else {
              result[0] += -0.15037241184248842;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        result[0] += -0.12061522215534576;
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.215408444404602495) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.03492136827448369;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.07461984957669558;
                } else {
                  result[0] += 0.00940437373621428;
                }
              }
            } else {
              result[0] += -0.09763885479852107;
            }
          } else {
            result[0] += -0.09668508949697946;
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.0012205771164568286;
          } else {
            result[0] += -0.03270261792240819;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.0678944205607587;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.08051929817822512;
          } else {
            result[0] += -0.0759529841628834;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.10263220948753332;
        } else {
          result[0] += 0.037694252706732545;
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.01161510406102662;
          } else {
            result[0] += -0.07810287321109244;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
            result[0] += -0.07475225803245114;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                    result[0] += -0.03665301326595465;
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                        result[0] += 0.04133337482821097;
                      } else {
                        result[0] += -0.026907610427818775;
                      }
                    } else {
                      result[0] += -0.047077641422996475;
                    }
                  }
                } else {
                  result[0] += -0.06331421052846264;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.013930446089090923;
                  } else {
                    result[0] += -0.08298657773400803;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    result[0] += -0.04836075382974732;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.20590913295745894) ) ) {
                            result[0] += 0.0617034912874013;
                          } else {
                            result[0] += -0.09038468280588662;
                          }
                        } else {
                          result[0] += 0.08106087362001158;
                        }
                      } else {
                        result[0] += 0.08693316330817864;
                      }
                    } else {
                      result[0] += 0.019326171447708113;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                  result[0] += 0.05510505096185408;
                } else {
                  result[0] += -0.03941172613963128;
                }
              } else {
                result[0] += -0.01308516579608729;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                result[0] += 0.05798885197375643;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.10125797187663022;
                } else {
                  result[0] += -0.02578501605591685;
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.3452332964869385;
                      } else {
                        result[0] += -0.07998673890137323;
                      }
                    } else {
                      result[0] += 0.005929006342553996;
                    }
                  } else {
                    result[0] += -0.0969770489556837;
                  }
                } else {
                  result[0] += 0.012220639990348784;
                }
              } else {
                result[0] += 0.03343538484930157;
              }
            }
          } else {
            result[0] += -0.07989724371253938;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
                result[0] += -0.09583404225095078;
              } else {
                result[0] += -0.008698376835192543;
              }
            } else {
              result[0] += 0.056735947493125466;
            }
          } else {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.08340198788900846;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.056876840806967394;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.01681899916830054;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.22558160616352754;
                        } else {
                          result[0] += 0.7874924975823836;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.070054531097412998) ) ) {
                          result[0] += -0.09445066656285707;
                        } else {
                          result[0] += 0.23618851383769882;
                        }
                      }
                    } else {
                      result[0] += -0.19217494045121108;
                    }
                  }
                } else {
                  result[0] += -0.0588793306160366;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)10.50000000000000178) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            result[0] += 0.008142610761778815;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.05607847603428896;
              } else {
                result[0] += -0.1805212090708204;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.057867360008158224;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.025360061284431763;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += -0.03159399310807296;
                  } else {
                    result[0] += -0.0028345440979193697;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.011972678635061455;
          } else {
            result[0] += 0.05372179669842281;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.042435407638550693) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.04828434704126972;
              } else {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.04980235246931008;
                } else {
                  if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.08049122320194142;
                  } else {
                    result[0] += -0.011182982232872757;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)10.50000000000000178) ) ) {
                if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41211462020874201) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                          result[0] += -0.03090020682084503;
                        } else {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.004793265537604171;
                            } else {
                              result[0] += -0.07883299328204807;
                            }
                          } else {
                            result[0] += 0.016196232390833353;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                              result[0] += -0.1264007938507625;
                            } else {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.607751369476319248) ) ) {
                                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
                                  result[0] += -0.08823843940836927;
                                } else {
                                  result[0] += 0.10563231145491468;
                                }
                              } else {
                                result[0] += -0.035894005071465916;
                              }
                            }
                          } else {
                            result[0] += 0.0258929222030974;
                          }
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                                result[0] += -0.0733953275416952;
                              } else {
                                result[0] += 0.18118286567082909;
                              }
                            } else {
                              result[0] += 0.04778916486229612;
                            }
                          } else {
                            result[0] += -0.07120457798990049;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.004285889256105366;
                      } else {
                        result[0] += -0.05274686186857106;
                      }
                    }
                  } else {
                    result[0] += -0.015031933427504357;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                    result[0] += -0.019984510666535428;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.947818994522095615) ) ) {
                          result[0] += 0.0387929405449737;
                        } else {
                          result[0] += -0.05215922143027836;
                        }
                      } else {
                        result[0] += -0.0013024828364142406;
                      }
                    } else {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += 0.057590680814583475;
                        } else {
                          result[0] += -0.06660297122749514;
                        }
                      } else {
                        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += 0.04917747686571394;
                        } else {
                          result[0] += -0.004681687888594264;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.07118793007357319;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)13.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.07919300125412929;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += -0.050022871884326026;
                    } else {
                      result[0] += 0.04843868725077158;
                    }
                  } else {
                    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
                      result[0] += 0.03279839969241833;
                    } else {
                      result[0] += 0.0018380831031493815;
                    }
                  }
                }
              } else {
                result[0] += -0.0771949416348771;
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.5350331876101285;
                      } else {
                        result[0] += 0.07337885079070033;
                      }
                    } else {
                      result[0] += -0.02605362379477466;
                    }
                  } else {
                    result[0] += -0.06730017545202877;
                  }
                } else {
                  result[0] += -0.08417483777600843;
                }
              } else {
                result[0] += 0.011345061888343534;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.008388668267448659;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                  result[0] += 0.052005925287080305;
                } else {
                  result[0] += -0.003961361389630552;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                result[0] += -0.00047863018999196857;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.0814604011986425;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
                    result[0] += 0.09628933757310977;
                  } else {
                    result[0] += -0.05150452716758095;
                  }
                }
              }
            }
          } else {
            result[0] += 0.0023790768072615457;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.008635513213490592;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
            result[0] += 0.05433778437263236;
          } else {
            result[0] += -0.05426796930589002;
          }
        }
      } else {
        result[0] += -0.030572422557046664;
      }
    }
  } else {
    result[0] += 0.012010377072883839;
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)6.500000000000000888) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.024464378467162182;
                } else {
                  result[0] += -0.012769644403734907;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.04700071577051396;
                } else {
                  result[0] += -0.16599978865131274;
                }
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += 0.008074916472905607;
              } else {
                result[0] += 0.05999857730870315;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.011800416938901498;
            } else {
              result[0] += 0.05181823499379628;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)6.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                result[0] += -0.0008619769910671306;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.0006251958987356017;
                    } else {
                      result[0] += -0.02390202062537221;
                    }
                  } else {
                    result[0] += -0.033073400673757254;
                  }
                } else {
                  result[0] += -0.03537838932514137;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.025328218001391442;
              } else {
                result[0] += 0.004793177313783461;
              }
            }
          } else {
            result[0] += 0.00015610909203321778;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.172047138214112216) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.780340790748596635) ) ) {
              result[0] += -0.03886888037882602;
            } else {
              result[0] += 0.0590401118019301;
            }
          } else {
            result[0] += 0.18356519586358688;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                  result[0] += 0.03806232515459414;
                } else {
                  result[0] += -0.12550431454209535;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  result[0] += -0.07401573082500344;
                } else {
                  result[0] += 0.0631570561430219;
                }
              }
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.00515892349392328;
              } else {
                result[0] += 0.07004012784200074;
              }
            }
          } else {
            result[0] += 0.07444105893432944;
          }
        }
      }
    } else {
      result[0] += -0.012676912967937487;
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.007656015332841422;
        } else {
          result[0] += 0.09043910711693189;
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
            result[0] += 0.08456722464849817;
          } else {
            result[0] += -0.07202189418678855;
          }
        } else {
          result[0] += 0.03442506247567156;
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.008352314476874867;
          } else {
            result[0] += -0.07664308682177468;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
            result[0] += -0.07383004846882966;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                    result[0] += 0.032729669262690095;
                  } else {
                    result[0] += -0.03242181474763844;
                  }
                } else {
                  result[0] += -0.03565901200851971;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.484580039978028232) ) ) {
                  result[0] += -0.046134789221763145;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += -0.10705891330179391;
                      } else {
                        result[0] += -0.009140513675224452;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += -0.09142276281359338;
                        } else {
                          result[0] += 5.183730859034999e-05;
                        }
                      } else {
                        result[0] += 0.03198024692743809;
                      }
                    }
                  } else {
                    result[0] += 0.02879520942661659;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.450390577316285068) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                  result[0] += 0.05214722105124489;
                } else {
                  result[0] += -0.03556697985077185;
                }
              } else {
                result[0] += -0.009319702643814484;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.042966282699082425;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.040745084705624025;
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)12.50000000000000178) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                      result[0] += -0.0026491725392218213;
                    } else {
                      result[0] += -0.0983565300337536;
                    }
                  } else {
                    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.07384627978854821;
                    } else {
                      result[0] += 0.006154665924080773;
                    }
                  }
                } else {
                  result[0] += 0.037150603316190994;
                }
              } else {
                result[0] += -0.06552342383610517;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                result[0] += -0.015936727361622063;
              } else {
                result[0] += -0.06445297161791919;
              }
            } else {
              result[0] += -0.07615575173098582;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.05547958452936717;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.05188953573762296;
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.05520588078428397;
                } else {
                  result[0] += -0.05327328991962728;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.008556070516861554;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                result[0] += -0.06805061039314392;
              } else {
                result[0] += 0.08203627344963685;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.719506263732911933) ) ) {
              result[0] += 0.0010467081727958745;
            } else {
              result[0] += 0.039398937806966884;
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.023782786897834623;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.060294389724732333) ) ) {
              result[0] += -0.028897771897458138;
            } else {
              result[0] += -0.0015878452473451093;
            }
          }
        }
      } else {
        result[0] += -0.05914774869057298;
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += 0.0028850908514968833;
                    } else {
                      result[0] += 0.039415842170317975;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += 0.02669996527559973;
                    } else {
                      result[0] += -0.05833228592434658;
                    }
                  }
                } else {
                  result[0] += -0.0416628860304184;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += -0.09843157135576555;
                  } else {
                    result[0] += 0.030724392895070996;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.08746278342228424;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += 0.05744645873678135;
                        } else {
                          result[0] += -0.11197967726725239;
                        }
                      }
                    } else {
                      result[0] += 0.017187304958961664;
                    }
                  } else {
                    result[0] += 0.03180540753011132;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.34969997406006037) ) ) {
                result[0] += -0.1136955344865161;
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.003646550348730803;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0018830053388898989;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += 0.022134862783003584;
                        } else {
                          result[0] += -0.06110246734970194;
                        }
                      } else {
                        result[0] += -0.06725546039648854;
                      }
                    } else {
                      result[0] += 0.002999721474740842;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.97438240051269709) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += -0.03054205244396756;
                } else {
                  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.03923635742953464;
                  } else {
                    result[0] += -0.09098845301597538;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  result[0] += 0.016039419829308437;
                } else {
                  result[0] += -0.03388113165177899;
                }
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.013168050769502008;
              } else {
                result[0] += -0.041132101555470035;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.008210383403430667;
              } else {
                result[0] += 0.01870460777749682;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.022752849533770908;
                } else {
                  result[0] += -0.08087962803413942;
                }
              } else {
                result[0] += -0.07314605274230013;
              }
            }
          } else {
            result[0] += 0.0017291750629108513;
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += -0.046919596266754036;
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.07923880506221176;
            } else {
              result[0] += 0.002144075390502141;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.004405573096458892;
            } else {
              result[0] += 0.05742547862455796;
            }
          } else {
            result[0] += -0.0211828446690062;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.08407610621156915;
      } else {
        result[0] += 0.005759322105556556;
      }
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.08814752251604327;
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.07696414072758594;
            } else {
              result[0] += -0.07582916870339997;
            }
          } else {
            result[0] += 0.03549961931881827;
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.08418565325584992;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.0694505662059182;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.12744418064873533;
                } else {
                  result[0] += 0.006970562015100091;
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += 0.040738911779730484;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                    result[0] += 0.04712157507748484;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.284418344497681552) ) ) {
                      result[0] += 0.007107902758820159;
                    } else {
                      result[0] += -0.06584176493463467;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.053283343884172774;
              } else {
                result[0] += 0.004934751112611308;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.238668441772461382) ) ) {
            result[0] += 0.0009156129344304702;
          } else {
            result[0] += -0.05519370175706695;
          }
        } else {
          result[0] += 0.021024927834888893;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.02590060811086313;
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.970608234405518466) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.0007934579931601203;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
                              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                  result[0] += 0.027771709287117252;
                                } else {
                                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                                    result[0] += -0.11311505625226739;
                                  } else {
                                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                                        result[0] += -0.0214994257305253;
                                      } else {
                                        result[0] += 0.6542055041058876;
                                      }
                                    } else {
                                      result[0] += -0.017579757801773;
                                    }
                                  }
                                }
                              } else {
                                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                  result[0] += -0.021371064958889463;
                                } else {
                                  result[0] += 0.2547745327667462;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                                result[0] += 0.022781694759229473;
                              } else {
                                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += -0.09372154716056359;
                                } else {
                                  result[0] += 0.0885929612425377;
                                }
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                                result[0] += -0.0019160506657438761;
                              } else {
                                result[0] += -0.12574648699273736;
                              }
                            } else {
                              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                                  result[0] += -0.04686232998726661;
                                } else {
                                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                                    result[0] += -0.00597281432744178;
                                  } else {
                                    result[0] += 0.09663115160305621;
                                  }
                                }
                              } else {
                                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                                  result[0] += 0.06890068089536606;
                                } else {
                                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                                    result[0] += -0.08032340356195321;
                                  } else {
                                    result[0] += -0.0047316031463997925;
                                  }
                                }
                              }
                            }
                          }
                        }
                      } else {
                        result[0] += -0.01677272153100737;
                      }
                    } else {
                      result[0] += -0.0833140977504388;
                    }
                  }
                } else {
                  result[0] += 0.07413906811511244;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.005089125836948903;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += -0.09158890232452299;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += 0.003054846709861467;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.06003296469943556;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                            result[0] += -0.024665911785665694;
                          } else {
                            result[0] += 0.07580491951667079;
                          }
                        }
                      } else {
                        result[0] += -0.00048184126155386905;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.042435407638550693) ) ) {
                result[0] += 0.011721503132744738;
              } else {
                result[0] += -0.07154001617476606;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
              if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)19.50000000000000355) ) ) {
                result[0] += -0.04130524812783368;
              } else {
                result[0] += 0.019937213847730676;
              }
            } else {
              result[0] += 0.0031441746806486824;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += -0.009713975610185263;
            } else {
              result[0] += -0.06454042814504855;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.585059762001038042) ) ) {
              result[0] += 0.0005442824802614511;
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.032993866587129335;
                } else {
                  result[0] += -0.04014518167996752;
                }
              } else {
                result[0] += 0.16188982896819398;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
          result[0] += -0.05518186813291799;
        } else {
          result[0] += 0.10389825766271013;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.012324577929725255;
          } else {
            result[0] += -0.038704313397737465;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.005732712566636901;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)104.0000000000000142) ) ) {
                result[0] += 0.06200286557181089;
              } else {
                result[0] += -0.0018972190179762758;
              }
            }
          } else {
            result[0] += -0.042166640434282275;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -0.05883832902050911;
    } else {
      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.0877710788426464;
        } else {
          result[0] += 0.025033381586453286;
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.0751734037595495;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.06627905539986763;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.595119953155518466) ) ) {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += 0.03676787605670378;
                } else {
                  result[0] += -0.017918227372811187;
                }
              } else {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.11718127921896895;
                } else {
                  result[0] += -0.0004694423457566889;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.0521348967481038;
              } else {
                result[0] += 0.004874272739395793;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
          result[0] += 0.004643560886746756;
        } else {
          result[0] += 0.023696416046007533;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                result[0] += -0.14264833779053543;
              } else {
                result[0] += 0.016260121423382925;
              }
            } else {
              result[0] += 0.10387500284844238;
            }
          } else {
            result[0] += -6.053990124326341e-05;
          }
        } else {
          result[0] += -0.08964403840920245;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.0019325380425465827;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
            result[0] += -0.05036211121584888;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += 0.02307964080954703;
              } else {
                result[0] += -0.03547662869257052;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.08218734802724202;
              } else {
                result[0] += 0.0036380076714118068;
              }
            }
          }
        }
      } else {
        result[0] += -0.0009033044962425676;
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.0010092842005423204;
        } else {
          result[0] += 0.08327911463339038;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.0711852346312377;
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.06530610515890498;
          } else {
            result[0] += 0.027959354032606083;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
            result[0] += -0.07615953412629353;
          } else {
            result[0] += 0.005846932892003071;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.056806643933558;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.010100623880319945;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                    result[0] += 0.029768350518723116;
                  } else {
                    result[0] += 0.12406319005878036;
                  }
                }
              } else {
                result[0] += -0.10797335979092203;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.050335801514777324;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.092913672213154;
                    } else {
                      result[0] += 0.04118178673258849;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                      result[0] += -0.1308628570520352;
                    } else {
                      result[0] += -0.018621798758397617;
                    }
                  }
                }
              } else {
                result[0] += 0.07786769975510097;
              }
            } else {
              result[0] += 0.014643454716354401;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += -0.026184327643012845;
          } else {
            result[0] += -0.08960518838076671;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                      result[0] += 0.07331321501861597;
                    } else {
                      result[0] += -0.02035566935417952;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                          result[0] += 0.34093749694319525;
                        } else {
                          result[0] += -0.09716587767634616;
                        }
                      } else {
                        result[0] += -0.06371515749641096;
                      }
                    } else {
                      result[0] += 0.009011156977121597;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.040600554059311886;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                      if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
                        result[0] += 0.5801841837667449;
                      } else {
                        result[0] += 0.09005468926315825;
                      }
                    } else {
                      result[0] += -0.006878944665928118;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)16.50000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                    result[0] += -0.10066936953467341;
                  } else {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)11.50000000000000178) ) ) {
                        result[0] += -0.005171511774998985;
                      } else {
                        result[0] += 0.11257354763802124;
                      }
                    } else {
                      result[0] += -0.06808789518248973;
                    }
                  }
                } else {
                  result[0] += 0.005316610755692237;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
                result[0] += 0.039041071073026234;
              } else {
                result[0] += -0.08065193689111921;
              }
            }
          } else {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
              result[0] += -0.06140264507351245;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += 0.07046921286722917;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.07956376225452079;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.033759364925748964;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                            result[0] += -0.002128473694457459;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 1.2420775601939917;
                            } else {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += -0.0740373134949191;
                              } else {
                                result[0] += 0.7580353129146454;
                              }
                            }
                          }
                        }
                      } else {
                        result[0] += -0.10261409355144845;
                      }
                    } else {
                      result[0] += 0.16907041288632021;
                    }
                  }
                } else {
                  result[0] += -0.049109911764582945;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
        result[0] += 0.002393125692630541;
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.004670691662748839;
          } else {
            result[0] += 0.05290498413477729;
          }
        } else {
          result[0] += 0.04714885902165157;
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.05690380948368412;
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.08013266115566622;
                } else {
                  result[0] += -0.031703406467364546;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)11.50000000000000178) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.05844522876292642;
                    } else {
                      result[0] += -0.0002293325347935338;
                    }
                  } else {
                    result[0] += -0.05354597599572728;
                  }
                } else {
                  if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)26.50000000000000355) ) ) {
                    result[0] += 0.06956321364644533;
                  } else {
                    result[0] += -0.04334812470096405;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.006190449126409419;
              } else {
                result[0] += -0.05437598135273555;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.947818994522095615) ) ) {
                  result[0] += 0.013984119104533906;
                } else {
                  result[0] += -0.07097217175726837;
                }
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.0072774178142204285;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                      result[0] += 0.030120856154890404;
                    } else {
                      result[0] += -0.032166647713668055;
                    }
                  } else {
                    result[0] += -0.06024380816457919;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.994492053985595925) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                result[0] += -0.009606788705819838;
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += 1.618726201330915;
                } else {
                  result[0] += 0.08520279912735723;
                }
              }
            } else {
              result[0] += -0.06184115057934517;
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.56849193572998225) ) ) {
                      result[0] += 0.05481702701780131;
                    } else {
                      result[0] += -0.0007806538851105358;
                    }
                  } else {
                    result[0] += 0.09159262174658489;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.172047138214112216) ) ) {
                    result[0] += 0.015808548738355733;
                  } else {
                    result[0] += -0.0876829736314544;
                  }
                }
              } else {
                result[0] += -0.06555214235292658;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += -0.00014392852594624734;
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.02616764771220631;
                  } else {
                    result[0] += -0.07144797799592499;
                  }
                } else {
                  result[0] += -0.07183691081522807;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.0070328062203484434;
          } else {
            result[0] += -0.038848827591316276;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.023463344327107023;
            } else {
              result[0] += 0.04190283052397725;
            }
          } else {
            result[0] += -0.03831371167503279;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.07843610562931368;
      } else {
        result[0] += 0.011241756202166572;
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.1003297648137621;
          } else {
            result[0] += -0.005610555513938359;
          }
        } else {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += 0.08725665011908247;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.038285990591724654;
              } else {
                result[0] += -0.08188593859334675;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
              result[0] += 0.035869388762702806;
            } else {
              result[0] += -0.03293249965335681;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.08486783440345291;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += -0.032671736177014195;
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += 0.14522768662670918;
              } else {
                result[0] += -0.04489902118525526;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.09006836202431504;
          } else {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += 0.0036334032378223416;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.481347560882569248) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += -0.10089993459844826;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.208071470260621005) ) ) {
                          result[0] += 1.0430401576397628;
                        } else {
                          result[0] += 2.318580066491224;
                        }
                      }
                    } else {
                      result[0] += -0.10537211358620846;
                    }
                  } else {
                    result[0] += -0.09737625202367917;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.050622992696493264;
                } else {
                  result[0] += 0.05379253891514851;
                }
              }
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                result[0] += -0.004430309917665844;
              } else {
                result[0] += 0.014716918841580693;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)7.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.607751369476319248) ) ) {
        result[0] += 0.002486580117451883;
      } else {
        result[0] += 0.02324636546134991;
      }
    } else {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)14.50000000000000178) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06340596190804729;
            } else {
              if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.06501550010840848;
                } else {
                  result[0] += -0.050349123914143534;
                }
              } else {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += -0.036589421688730375;
                } else {
                  result[0] += 0.04103221975179007;
                }
              }
            }
          } else {
            result[0] += -0.004053213971229997;
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.894675970077515537) ) ) {
              result[0] += 0.049718737421512754;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.04764122628487675;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.02604460716247603) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += 0.17335512550092055;
                    } else {
                      result[0] += -0.0019333857465134545;
                    }
                  } else {
                    result[0] += -0.05751215033209019;
                  }
                }
              } else {
                result[0] += -0.07655985154005988;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.297559976577759233) ) ) {
              result[0] += -0.06402220231732621;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.35853882918872126;
                } else {
                  result[0] += 0.03220545839050742;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)40.00000000000000711) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += 0.04517177003510238;
                    } else {
                      result[0] += 0.010666228607352697;
                    }
                  } else {
                    result[0] += -0.06569723055820274;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += -0.013890466778230116;
                  } else {
                    result[0] += 0.0037378891758289533;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
              result[0] += -0.12419626993506985;
            } else {
              result[0] += 0.09119677028939738;
            }
          } else {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.1939352907945303;
            } else {
              result[0] += 0.016351304594469086;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += 0.07799204971724998;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.161602735519410068) ) ) {
              result[0] += 0.04876953218068428;
            } else {
              result[0] += -0.08904454389939649;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += 0.05596933998598228;
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
          result[0] += -0.07275199810237952;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
            result[0] += 0.0693246058544487;
          } else {
            if ( LIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)18.50000000000000355) ) ) {
              result[0] += 0.026808679410954134;
            } else {
              result[0] += -0.03561208238754381;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
            result[0] += -0.07449782697704735;
          } else {
            result[0] += 0.003915574492090345;
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.04529513455726747;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                  result[0] += -0.04213808878975337;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
                    result[0] += 0.030942400976902748;
                  } else {
                    result[0] += -0.038893506462319094;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.051912069320679599) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                  result[0] += -0.06815934471545801;
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.03670689319008462;
                  } else {
                    result[0] += 0.04023882324712498;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                  result[0] += -0.013097436693062604;
                } else {
                  result[0] += 0.029730069517643756;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
              result[0] += 0.03296866767049283;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.005519303105848951;
                } else {
                  result[0] += -0.06223733686633509;
                }
              } else {
                result[0] += 0.025548532622008832;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.05356444829809214;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += 0.03598086288823512;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.0005039779497129954;
                } else {
                  result[0] += -0.03650219722332285;
                }
              } else {
                result[0] += 0.027117990931896954;
              }
            }
          } else {
            if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)15.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.02674811212080949;
                } else {
                  result[0] += 0.07891377119415757;
                }
              } else {
                result[0] += -0.05789648034574583;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += 0.050999133554278625;
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.031248185659735986;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.070054531097412998) ) ) {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.0957626836103414;
                      } else {
                        result[0] += -0.03334610630526668;
                      }
                    } else {
                      result[0] += 0.1496412370021304;
                    }
                  } else {
                    result[0] += -0.1023156823038435;
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
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.076962471008301669) ) ) {
                result[0] += -0.0019902026368546133;
              } else {
                result[0] += -0.08375058059050008;
              }
            } else {
              result[0] += -0.09737678537034948;
            }
          } else {
            result[0] += 0.0043472161517234395;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.25930547714233576) ) ) {
            result[0] += 0.005376702888087962;
          } else {
            result[0] += 0.04322309970083526;
          }
        }
      } else {
        result[0] += 0.019662984515948785;
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.142747402191162998) ) ) {
                    result[0] += 0.009503570423628137;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += 0.023089353720777014;
                    } else {
                      result[0] += -0.05832448048304664;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.172047138214112216) ) ) {
                    result[0] += -0.09434075004644472;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.09492412833399867;
                    } else {
                      result[0] += -0.012937813956287078;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += -0.09445653045285642;
                  } else {
                    result[0] += 0.02726224502521396;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.651049375534058505) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += -0.07290918036696685;
                      } else {
                        result[0] += 0.013440840886172324;
                      }
                    } else {
                      result[0] += -0.09454600881069597;
                    }
                  } else {
                    result[0] += 0.028089875293777974;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.34969997406006037) ) ) {
                result[0] += -0.10980099681596232;
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.02299052217878034;
                  } else {
                    result[0] += 0.010478723455624374;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += -0.10723224138346968;
                      } else {
                        result[0] += -0.0034652120755874513;
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.011033561755521375;
                      } else {
                        result[0] += 0.07652920435106551;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.00655247080300472;
                    } else {
                      if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.06514745454960535;
                      } else {
                        result[0] += 0.0062565819770866865;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.01053571701049982) ) ) {
                      result[0] += -0.09757580734339283;
                    } else {
                      result[0] += -0.01776877355140653;
                    }
                  } else {
                    result[0] += 0.018314329542234598;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.014822515504367452;
                  } else {
                    result[0] += -0.07750929091455964;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                  result[0] += 0.014264549483563246;
                } else {
                  result[0] += -0.03334316721252036;
                }
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.012832116886302156;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.053809675781541035;
                } else {
                  result[0] += -0.0028589703624910894;
                }
              }
            }
          }
        } else {
          result[0] += -0.0004819773337443393;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.00017627130145171172;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
            result[0] += 0.035414089989466636;
          } else {
            result[0] += -0.05896540416699836;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.07598600892831227;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
          result[0] += 0.07493984292237985;
        } else {
          result[0] += -0.06881378933302465;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.08901836895083835;
          } else {
            result[0] += -0.009285170606266109;
          }
        } else {
          result[0] += 0.024030472672399834;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.08961829865736627;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.06034930348921;
            } else {
              result[0] += 0.043932009695686856;
            }
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.07769479947849432;
          } else {
            if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.03909470730027027;
                  } else {
                    result[0] += -0.04825976526307418;
                  }
                } else {
                  result[0] += 0.02946461745164848;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.04981442993934301;
                } else {
                  result[0] += 0.04941403341581885;
                }
              }
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                result[0] += -0.004405986613002147;
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.363266706466675693) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.020261231956104773;
                    } else {
                      result[0] += 0.02842538563846006;
                    }
                  } else {
                    result[0] += 0.07703824930494652;
                  }
                } else {
                  result[0] += -0.012155318003857933;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += -0.0905650546078756;
              } else {
                result[0] += 0.026326119504658033;
              }
            } else {
              result[0] += -0.0001667482706108197;
            }
          } else {
            result[0] += -0.06957767065217602;
          }
        } else {
          result[0] += 0.008063181002333343;
        }
      } else {
        result[0] += 0.022346097204843346;
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.00402034192530963;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                      result[0] += 0.026479398926641435;
                    } else {
                      result[0] += -0.08390831624927676;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.005405997604106306;
                    } else {
                      result[0] += -0.06652268195747585;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.978769779205324042) ) ) {
                    result[0] += 0.01470793746259341;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.516936540603638583) ) ) {
                      result[0] += -0.042644548189593935;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.048644404945977764;
                      } else {
                        result[0] += -0.042381766112839836;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                    result[0] += 0.047356093416343496;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.484580039978028232) ) ) {
                      result[0] += 0.01927512173539641;
                    } else {
                      result[0] += -0.03154458691985503;
                    }
                  }
                } else {
                  result[0] += -0.077294202920266;
                }
              }
            }
          } else {
            result[0] += -0.04134901969989161;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.11666350642265716;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0018521007116077551;
                  } else {
                    result[0] += 0.13065917138392744;
                  }
                } else {
                  result[0] += -0.03436286513405527;
                }
              }
            } else {
              result[0] += -0.00576618736164794;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              result[0] += -0.010594553708220177;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.06530057644592072;
              } else {
                result[0] += -0.0155288711799946;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += 0.051535663781521024;
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.06646315206795951;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.005292551236358529;
                  } else {
                    result[0] += 0.03214536650555264;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.003982548852161766;
                    } else {
                      result[0] += 0.04684227885540149;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.607751369476319248) ) ) {
                        result[0] += 0.020346173152400965;
                      } else {
                        result[0] += -0.0582383703809142;
                      }
                    } else {
                      result[0] += -0.06824776761510898;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.05297899612789636;
                } else {
                  result[0] += 0.08111919241696301;
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.051912069320679599) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.000000000000000888) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.008059641890084781;
                      } else {
                        result[0] += 0.04292495361480896;
                      }
                    } else {
                      result[0] += -0.008112249996009463;
                    }
                  } else {
                    result[0] += 0.0016068667245805184;
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.07346926236431002;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                        result[0] += 0.12896624650551508;
                      } else {
                        result[0] += -0.09609484164997802;
                      }
                    } else {
                      result[0] += 0.011689591872001042;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.06622938591775586;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.062379319666604774;
                    } else {
                      result[0] += -0.012152643063762843;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.844439744949341042) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.06871663747546532;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.015591661862722081;
                          } else {
                            result[0] += 0.11791963161299264;
                          }
                        } else {
                          result[0] += -0.0496125893601464;
                        }
                      }
                    } else {
                      result[0] += 0.018130133703790526;
                    }
                  } else {
                    result[0] += 0.0462738638777544;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.07427624441744995;
      } else {
        result[0] += 0.02487452924125431;
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.02708939479417497;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += -0.08856952477988059;
          } else {
            result[0] += 0.0034308194994233706;
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.06705601472395953;
          } else {
            result[0] += 0.006069190012960909;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              result[0] += -0.005027238784286476;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.04723724583845232;
              } else {
                result[0] += -0.16917447360800256;
              }
            }
          } else {
            result[0] += 0.007320048921844806;
          }
        } else {
          result[0] += 0.026281997372922555;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.135017871856690341) ) ) {
          result[0] += -0.021158165853120928;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              result[0] += 0.02847260941670556;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                result[0] += -0.007247089277324938;
              } else {
                result[0] += -0.21186137318390716;
              }
            }
          } else {
            result[0] += 0.08364047470505898;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)80.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += 0.2899229529207915;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)120.0000000000000142) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)20.00000000000000355) ) ) {
                    result[0] += 0.00033853744743065565;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.232423543930054599) ) ) {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                        result[0] += 0.18861507636094332;
                      } else {
                        result[0] += -0.0009930518651013564;
                      }
                    } else {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.0489771890562105;
                        } else {
                          result[0] += -0.0751272016992676;
                        }
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)56.00000000000000711) ) ) {
                            result[0] += -0.08289373955121974;
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.09029597586956102;
                            } else {
                              result[0] += -0.02653822568957506;
                            }
                          }
                        } else {
                          result[0] += 0.014377485748429975;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.12322243488783274;
                  } else {
                    result[0] += 0.01090319798032095;
                  }
                }
              } else {
                result[0] += -0.14127921357330195;
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += -0.02500127913390468;
                } else {
                  result[0] += -0.004885394313932284;
                }
              } else {
                result[0] += -0.08039997932539912;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.04583643577384525;
                    } else {
                      result[0] += 0.060278332159554075;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.938867926597595659) ) ) {
                        result[0] += -0.1851280657408092;
                      } else {
                        result[0] += -0.02488011232608866;
                      }
                    } else {
                      result[0] += 0.02711144353574982;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.87548160552978693) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.302512168884278232) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                          result[0] += -0.031410418834511095;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.01777522076416615;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                              result[0] += 0.0783177470097854;
                            } else {
                              result[0] += -0.06267083641323781;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.01269407602126068;
                      }
                    } else {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.07729864120483576) ) ) {
                          result[0] += 0.018418608823209604;
                        } else {
                          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.025330357123726097;
                          } else {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                                result[0] += 0.11456268334393604;
                              } else {
                                result[0] += -0.022746541376879894;
                              }
                            } else {
                              result[0] += 0.07238250718826751;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                            result[0] += -0.1331362497568581;
                          } else {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                              result[0] += -0.0829850712890735;
                            } else {
                              result[0] += -0.029687912185956518;
                            }
                          }
                        } else {
                          result[0] += 0.05375638956339664;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.052063072521469925;
                    } else {
                      result[0] += -0.08430715234373067;
                    }
                  }
                }
              } else {
                result[0] += -0.04068682720325376;
              }
            }
          }
        } else {
          result[0] += -0.00017789206131250805;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.00513115733276562;
          } else {
            result[0] += -0.039612644421722146;
          }
        } else {
          result[0] += 0.026635682316678146;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.023550365800339392;
        } else {
          result[0] += 0.0285856921179674;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
          result[0] += 0.07280239322494522;
        } else {
          result[0] += 0.02682027791028061;
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.04671065860194523;
      } else {
        if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.020238653490516054;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.046916009606770774;
          } else {
            if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.05268703646076234;
            } else {
              if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)17.50000000000000355) ) ) {
                result[0] += -0.00599181062982772;
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.02346687980389967;
                } else {
                  result[0] += -0.009835889183120753;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += 0.006334721291597634;
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.11188630859680615;
            } else {
              result[0] += -0.01146549702938713;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.05194581793189702;
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.020042144321066855;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.03694224325217499;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.607751369476319248) ) ) {
                  result[0] += -0.007901235024325503;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.15096221268814394;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.08693193656890848;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += 0.10984661234029983;
                      } else {
                        result[0] += -0.03971911600696741;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.03250877466177947;
            } else {
              result[0] += 0.09394919918107342;
            }
          } else {
            result[0] += -0.1008276672027437;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.25930547714233576) ) ) {
                result[0] += 0.016729849733135835;
              } else {
                result[0] += 0.04684248917551192;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.17256754074449818;
              } else {
                result[0] += -0.002753321490337443;
              }
            }
          } else {
            result[0] += 0.09060466363489324;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)13.50000000000000178) ) ) {
        result[0] += -0.002502766203487691;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.651049375534058505) ) ) {
              result[0] += 0.01010668039419374;
            } else {
              result[0] += -0.05519815903957301;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.014290796470615869;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.05184301334575714;
              } else {
                result[0] += -0.05465985590135047;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.024457571738553716;
              } else {
                result[0] += -0.08060708996116109;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.07538076025102175;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                      result[0] += -0.022391189486394687;
                    } else {
                      result[0] += 0.24263768532504437;
                    }
                  } else {
                    result[0] += 0.06551510747231308;
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.03512278147730223;
                  } else {
                    result[0] += 0.03179660010528645;
                  }
                }
              }
            }
          } else {
            result[0] += -0.03887879155468446;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)9.500000000000001776) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)13.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.827801465988160068) ) ) {
            result[0] += 0.09367436965830726;
          } else {
            result[0] += 0.016683888570487953;
          }
        } else {
          result[0] += -0.01869245464311811;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.694163918495178667) ) ) {
          result[0] += 0.08193282192546034;
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)6.500000000000000888) ) ) {
            result[0] += -0.10478537802555639;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
              result[0] += 0.06969244061356876;
            } else {
              result[0] += 0.012818409130507906;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.484580039978028232) ) ) {
            result[0] += 0.07980749264089251;
          } else {
            result[0] += -0.07602337897831014;
          }
        } else {
          result[0] += -0.07670991914252356;
        }
      } else {
        if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
            if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)12.50000000000000178) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.09830135483527928;
              } else {
                result[0] += -0.08792322211327702;
              }
            } else {
              result[0] += 0.07117683485158095;
            }
          } else {
            result[0] += -0.0919639871736678;
          }
        } else {
          if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.06749053889311925;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( LIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)9.500000000000001776) ) ) {
                    result[0] += 0.05232037482654611;
                  } else {
                    result[0] += 0.17660379013629407;
                  }
                } else {
                  result[0] += -0.06995876009736533;
                }
              } else {
                if ( UNLIKELY( !(data[47].missing != -1) || (data[47].fvalue <= (double)11.50000000000000178) ) ) {
                  result[0] += 0.025418064885582375;
                } else {
                  result[0] += -0.011541057539143505;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.08945078997905546;
                } else {
                  result[0] += -0.004663114330091019;
                }
              } else {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.95478391647339045) ) ) {
                    result[0] += -0.038061061366234515;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.11367731889244925;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.11431867204562406;
                      } else {
                        result[0] += 0.012796346282946878;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.830332040786744052) ) ) {
                    result[0] += 0.00947182546491522;
                  } else {
                    if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.038594474230191335;
                    } else {
                      result[0] += -0.07390727812874723;
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
  if ( LIKELY( !(data[51].missing != -1) || (data[51].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += 0.006278641090830456;
          } else {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.11199764462875693;
            } else {
              result[0] += -0.011523805018686639;
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.019161323557469006;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.924581527709961826) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.04971636143022865;
              } else {
                result[0] += -0.005224120245566611;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.215905904769898349) ) ) {
                result[0] += -0.2274231282649044;
              } else {
                result[0] += 0.04079625332574092;
              }
            }
          }
        }
      } else {
        result[0] += 0.02537732758544844;
      }
    } else {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)2.221325635910034624) ) ) {
            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.184516429901124823) ) ) {
                  result[0] += -0.05867033823309014;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                      result[0] += -0.10937513335957455;
                    } else {
                      result[0] += -0.00893277405468961;
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.05849283506679377;
                    } else {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.018048929014396;
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.0808203366950749;
                        } else {
                          result[0] += 0.025272606246882795;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                      result[0] += -0.005459751155812286;
                    } else {
                      result[0] += -0.021774251491646586;
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.12072100502824397;
                    } else {
                      result[0] += -0.0010643150718223616;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.10011926223580488;
                  } else {
                    result[0] += 0.07346558680345107;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.516936540603638583) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.669892311096192294) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.354025125503540261) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.21891870391128254;
                      } else {
                        result[0] += -0.03430361552084837;
                      }
                    } else {
                      result[0] += 0.29753793282434454;
                    }
                  } else {
                    result[0] += -0.07344867810100074;
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.062089802754263124;
                    } else {
                      result[0] += -0.11262650897119297;
                    }
                  } else {
                    result[0] += -0.09005630240321366;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.172047138214112216) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                    result[0] += -0.03324758553387809;
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.027152171590496856;
                    } else {
                      result[0] += 0.18059414589425943;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.0027697110137829;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.27488834490509656;
                    } else {
                      result[0] += 0.04631022840373212;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.1473647323959981;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.994492053985595925) ) ) {
            result[0] += -0.11234892515923466;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.04636812210083185) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)2.221325635910034624) ) ) {
                result[0] += 0.00025843817621827776;
              } else {
                result[0] += -0.12961976272902428;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.0440915810148716;
              } else {
                result[0] += 0.01696025944461221;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.04871103816261285;
          } else {
            result[0] += -0.00037369857278466;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.040018744886924994;
            } else {
              result[0] += -0.03137057167198646;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                result[0] += 0.1459769938570458;
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.05854447575578353;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.021878093682256603;
                  } else {
                    result[0] += 0.1526099942640611;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.07965185955951591;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)10.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.08508603249120789;
                    } else {
                      result[0] += -0.0788017096644821;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.516936540603638583) ) ) {
                        result[0] += -0.08268364962446967;
                      } else {
                        result[0] += 0.02972176329390598;
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)72.00000000000001421) ) ) {
                        result[0] += 0.12884311579966087;
                      } else {
                        result[0] += 0.04431942358358108;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.08318330114652288;
                  } else {
                    result[0] += -0.24103738871462907;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[50].missing != -1) || (data[50].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.017476593912683542;
      } else {
        result[0] += -0.0721368735663302;
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.261864185333252841) ) ) {
          result[0] += 0.026544251781688566;
        } else {
          result[0] += -0.03196361636347658;
        }
      } else {
        result[0] += 0.004991533219649176;
      }
    }
  }
}

