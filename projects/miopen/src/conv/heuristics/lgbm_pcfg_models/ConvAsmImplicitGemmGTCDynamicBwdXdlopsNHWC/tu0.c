
#include "header.h"

void predict_unit0(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += 0.035375609581107796;
              } else {
                result[0] += -0.11255902025901264;
              }
            } else {
              result[0] += 0.06832493569200669;
            }
          } else {
            result[0] += 0.09722126363766624;
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.1369422555114019;
              } else {
                result[0] += 0.005511967882309324;
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.10012138795858597;
              } else {
                result[0] += 0.1735339957066918;
              }
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.13374424705078164;
                  } else {
                    result[0] += 0.004181236662426576;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.010087930069757233;
                  } else {
                    result[0] += 0.10727532261989953;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.06944006116772768;
                } else {
                  result[0] += 0.16775228221848;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.09169442071867807;
                } else {
                  result[0] += 0.030412480100414713;
                }
              } else {
                result[0] += 0.0828926598593816;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.10285831380502984;
          } else {
            result[0] += 0.0064981573535773315;
          }
        } else {
          result[0] += 0.04302222378192819;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += 0.11801818851762565;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.0033343950382871087;
              } else {
                result[0] += -0.12054940011368694;
              }
            } else {
              result[0] += 0.12654532472706578;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.09986695444264308;
                } else {
                  result[0] += 0.061541759965924175;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)234.5000000000000284) ) ) {
                  result[0] += 0.10651374652268449;
                } else {
                  result[0] += -0.036713720441806;
                }
              }
            } else {
              result[0] += -0.09706266404234265;
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.11541047029530727;
            } else {
              result[0] += -0.017597624994607125;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
          result[0] += -0.017027393771567725;
        } else {
          result[0] += -0.1531584645210173;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.740175724029542792) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
              result[0] += -0.01711802124449303;
            } else {
              result[0] += -0.10993724038069022;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.10469524611610992;
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                    result[0] += -0.08640903282425924;
                  } else {
                    result[0] += 0.07999598357012658;
                  }
                } else {
                  result[0] += -0.06162823056581034;
                }
              } else {
                result[0] += -0.09026838902186794;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
              result[0] += -0.052816123345672275;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.10555721360548752;
              } else {
                result[0] += -0.17010054655561482;
              }
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.034388780748901504;
              } else {
                result[0] += -0.05541911721256966;
              }
            } else {
              result[0] += -0.10115365684771543;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.053462417290664435;
          } else {
            result[0] += -0.14197072873929792;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += -0.11404542616254981;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                  result[0] += -0.1629027845202371;
                } else {
                  result[0] += -0.07780776360531613;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                  result[0] += 0.006629953690804303;
                } else {
                  result[0] += -0.08974373520205402;
                }
              }
            } else {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                    result[0] += -0.07834201584912387;
                  } else {
                    result[0] += 0.04247725388103171;
                  }
                } else {
                  result[0] += -0.0705798046015432;
                }
              } else {
                result[0] += -0.11275034687056794;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
          result[0] += 0.08396783608472909;
        } else {
          result[0] += -0.11644730670772174;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.740175724029542792) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.1603853243672801;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += -0.1011183798115192;
            } else {
              result[0] += 0.06976166486261412;
            }
          }
        } else {
          result[0] += -0.1791465613435486;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.1327942436844994;
              } else {
                result[0] += 0.04415885322257276;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                result[0] += 0.009674143296943957;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)89.50000000000001421) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.14086333056028552;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.13077218439752164;
                    } else {
                      result[0] += 0.10082778167958467;
                    }
                  }
                } else {
                  result[0] += 0.07075431604571862;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.03047949769084274;
            } else {
              result[0] += 0.03771697608591617;
            }
          }
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.07688866810872702;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += -0.14813072675328906;
                } else {
                  result[0] += 0.058594655476920346;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.06244103640278838;
                } else {
                  result[0] += -0.08802671390292406;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)85.50000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.549732685089113104) ) ) {
                    result[0] += -0.020254390400124472;
                  } else {
                    result[0] += 0.1466131178458053;
                  }
                } else {
                  result[0] += -0.0028658193348589443;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += -0.17792354106657637;
            } else {
              result[0] += -0.03113741798857594;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0908063817755268;
          } else {
            result[0] += -0.010651080447510059;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.14058845104775022;
            } else {
              result[0] += -0.06142512891882203;
            }
          } else {
            result[0] += 0.16302636713649718;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.133806212785436;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.052761459972552875;
                } else {
                  result[0] += -0.029161420819676872;
                }
              } else {
                result[0] += -0.07461502635866778;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.026612853243277952;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
                    result[0] += -0.003001893997141468;
                  } else {
                    result[0] += -0.111640938434753;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                  result[0] += -0.09250561786646799;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                      result[0] += 0.001731003721403756;
                    } else {
                      result[0] += 0.093302926750714;
                    }
                  } else {
                    result[0] += -0.028593574984449577;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                    result[0] += 0.01963907651471823;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)199.5000000000000284) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.1178914527789759;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.377930641174318183) ) ) {
                          result[0] += 0.11442081096931414;
                        } else {
                          result[0] += -0.00487907561989724;
                        }
                      }
                    } else {
                      result[0] += -0.029760567363436866;
                    }
                  }
                } else {
                  result[0] += -0.01054265789707647;
                }
              } else {
                result[0] += -0.13418821517718335;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.603528499603273261) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
              result[0] += 0.016934300531177796;
            } else {
              result[0] += -0.0444378850518689;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.039892339571404374;
            } else {
              result[0] += -0.11249219167309482;
            }
          }
        }
      } else {
        result[0] += -0.09534069946391645;
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.01002672489848246;
        } else {
          result[0] += -0.04715868768326997;
        }
      } else {
        result[0] += -0.1332745359938917;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.291543006896974433) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.07382394362086768;
            } else {
              result[0] += 0.023737922905027;
            }
          } else {
            result[0] += -0.07494866807468768;
          }
        } else {
          result[0] += -0.09790524313178627;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.11289940135787276;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.03780033970531828;
              } else {
                result[0] += 0.03127554784341865;
              }
            } else {
              result[0] += -0.07248020095632125;
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.1122699118272673;
            } else {
              result[0] += -0.15972887006454545;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.12597569208192005;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                result[0] += -0.09974689058413427;
              } else {
                result[0] += 0.06604915546320148;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
              result[0] += 0.08093186508960157;
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
                result[0] += -0.06854633059098357;
              } else {
                result[0] += 0.03926019158503891;
              }
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.08544633973820183;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.012136468190203836;
                } else {
                  result[0] += 0.05285965031538353;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.09995303164318438;
              } else {
                result[0] += 0.015353902616798115;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.749434947967529741) ) ) {
              result[0] += 0.05309516052640689;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
                result[0] += -0.06909182878460624;
              } else {
                result[0] += 0.10438813719369196;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.07198788928511461;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.04525324133626016;
              } else {
                result[0] += 0.12585230062901062;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                result[0] += 0.0412615466814451;
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.08780345192843134;
                } else {
                  result[0] += -0.021488495312809996;
                }
              }
            } else {
              result[0] += -0.07489354758460738;
            }
          } else {
            result[0] += -0.04144731187334358;
          }
        } else {
          result[0] += -0.010113136212975432;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.023842428903595017;
                  } else {
                    result[0] += 0.04585794843841463;
                  }
                } else {
                  result[0] += -0.07775004613090977;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.06730807868212889;
                  } else {
                    result[0] += 0.08526814934301068;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += -0.042265136719464046;
                  } else {
                    result[0] += 0.02827175371175386;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.0001580983449903799;
              } else {
                result[0] += 0.10460703224374586;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
              result[0] += -0.0009596472458005379;
            } else {
              result[0] += -0.06858942434003161;
            }
          }
        } else {
          result[0] += -0.10107368238325386;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.772694945335388628) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += -0.01558531153934066;
              } else {
                result[0] += -0.12314264755343521;
              }
            } else {
              result[0] += -0.1475337282432936;
            }
          } else {
            result[0] += -0.14285377048780062;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.05684076767516445;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.0019607261196212533;
            } else {
              result[0] += -0.14009007989568986;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.02618751391640538;
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                  result[0] += -0.12976473657803353;
                } else {
                  result[0] += 0.00562539669384079;
                }
              } else {
                result[0] += 0.0572636441780242;
              }
            } else {
              result[0] += -0.10337662933016596;
            }
          } else {
            result[0] += 0.036033052238423825;
          }
        }
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.09545049720213283;
          } else {
            result[0] += -0.006456111900901788;
          }
        } else {
          result[0] += -0.1188504516962468;
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.049745559692384589) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
                  result[0] += 0.010304671914423622;
                } else {
                  result[0] += -0.09702316542646794;
                }
              } else {
                result[0] += -0.08399287329868303;
              }
            } else {
              result[0] += -0.06863621402085078;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.06391135521036756;
            } else {
              result[0] += -0.14338158105047583;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.04488437683320077;
            } else {
              result[0] += -0.10655618401767242;
            }
          } else {
            result[0] += -0.12907511516590928;
          }
        }
      } else {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
          result[0] += -0.017141103050542023;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0928877939072883;
                } else {
                  result[0] += -0.030092613654279277;
                }
              } else {
                result[0] += -0.11140105546458264;
              }
            } else {
              result[0] += -0.13907058657798424;
            }
          } else {
            result[0] += -0.14133183039263555;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.020948775679305753;
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                result[0] += 0.05019340899756578;
              } else {
                result[0] += -0.06527899494923388;
              }
            } else {
              result[0] += -0.022954293648456716;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.897482156753540927) ) ) {
              result[0] += 0.05581162014010929;
            } else {
              result[0] += 0.09024222757418102;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
          result[0] += 0.07498019020400486;
        } else {
          if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.08223984367129165;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.10545404425428745;
              } else {
                result[0] += -0.0025964082881089682;
              }
            } else {
              result[0] += 0.1344960860800021;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.816582441329956943) ) ) {
                result[0] += -0.05673184604505027;
              } else {
                result[0] += 0.026870886817441483;
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.025106451064137038;
              } else {
                result[0] += 0.04542481370680166;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.05998358776674026;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                      result[0] += 0.023215978206915393;
                    } else {
                      result[0] += 0.07804089096636058;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.025569774828634075;
                  } else {
                    result[0] += -0.09774654133705442;
                  }
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.0727567192880516;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                      result[0] += 0.09421737525453289;
                    } else {
                      result[0] += 0.005411522110942366;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.015237415217117804;
                    } else {
                      result[0] += -0.1462476277453019;
                    }
                  } else {
                    if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.05409448845160624;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.04908161812126607;
                      } else {
                        result[0] += -0.06957127166811397;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.10443196519364813;
                } else {
                  result[0] += -0.005960175528628941;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                      result[0] += 0.08931545446845214;
                    } else {
                      result[0] += 0.034029027281554214;
                    }
                  } else {
                    result[0] += 0.002379512553714545;
                  }
                } else {
                  result[0] += -0.00343205394071413;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.09557946751264829;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0050544800381215555;
            } else {
              result[0] += -0.08801109889032586;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += -0.11488019349675548;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.04421089228489915;
          } else {
            result[0] += -0.08554369333323764;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.027707919484338137;
          } else {
            result[0] += 0.014493606200410791;
          }
        } else {
          result[0] += -0.08001030431537623;
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
          result[0] += -0.024730902489684542;
        } else {
          result[0] += -0.10854541531184639;
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.941534638404846635) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += 0.02218293795543061;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.030082833063371613;
                  } else {
                    result[0] += -0.11178709250373119;
                  }
                }
              } else {
                result[0] += 0.02246302198600225;
              }
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.006323256392861901;
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.539299011230469638) ) ) {
                  result[0] += -0.045716213119011605;
                } else {
                  result[0] += -0.10399429275343029;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08534226394526492;
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.07217136505493395;
                } else {
                  result[0] += -0.016678149823590036;
                }
              }
            } else {
              result[0] += -0.1293807686088234;
            }
          }
        } else {
          result[0] += -0.11926936684113941;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.11124575936277464;
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.08321382423037843;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.132848501205445224) ) ) {
                  result[0] += -0.00195738271760549;
                } else {
                  result[0] += -0.06949082589515843;
                }
              }
            }
          } else {
            result[0] += -0.14009523850013725;
          }
        } else {
          result[0] += -0.12829320859678198;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0887146538610919;
                } else {
                  result[0] += 0.03518641017171086;
                }
              } else {
                result[0] += 0.030244856117970925;
              }
            } else {
              if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.05041651591903602;
              } else {
                result[0] += 0.0023354802390273778;
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.017793843641608385;
            } else {
              result[0] += -0.07866254028945548;
            }
          }
        } else {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                  result[0] += 0.12235897073642826;
                } else {
                  result[0] += -0.022269283374010948;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.061206169280815764;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
                        result[0] += -0.09305546422665235;
                      } else {
                        result[0] += 0.032222814105300916;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
                      result[0] += 0.10013125897170597;
                    } else {
                      result[0] += 0.06302311021779942;
                    }
                  }
                } else {
                  result[0] += 0.004304307701801435;
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                    result[0] += 0.047322041552444176;
                  } else {
                    result[0] += 0.10064561085166401;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.08179332217611926;
                  } else {
                    result[0] += 0.11817138277051326;
                  }
                }
              } else {
                result[0] += 0.042143585572576636;
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)199.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.005601734236228936;
                  } else {
                    result[0] += -0.08817363501289704;
                  }
                } else {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06869940396737309;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += -0.10194110112453492;
                    } else {
                      result[0] += 0.01570801988151865;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.02841258338866015;
                } else {
                  result[0] += 0.08325164723839318;
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                  result[0] += -0.021481717745277226;
                } else {
                  result[0] += 0.036779725200767166;
                }
              } else {
                result[0] += -0.04479573757593798;
              }
            }
          }
        }
      } else {
        result[0] += -0.055231887736687316;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.0908014700629401;
            } else {
              result[0] += 0.023382424196560336;
            }
          } else {
            result[0] += -0.05626588045955754;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
            result[0] += 0.04482424079980217;
          } else {
            result[0] += -0.10172987515496308;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.01520729064941584) ) ) {
            result[0] += -0.017476172747457316;
          } else {
            result[0] += -0.08126193468578433;
          }
        } else {
          result[0] += -0.11544084171110208;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.03603886333969604;
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.023090040181086794;
            } else {
              result[0] += -0.08327151982534457;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.065111663619924;
          } else {
            result[0] += -0.13615100201209848;
          }
        }
      } else {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.033769557673927794;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.07918979081599382;
          } else {
            result[0] += -0.007421364428500258;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.740175724029542792) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += -0.014243814281352726;
            } else {
              result[0] += -0.11982122232946271;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.869741916656495029) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.023025227159017805;
                } else {
                  result[0] += -0.10478920231124991;
                }
              } else {
                result[0] += 0.03511655559898137;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += -0.0024051019141169807;
              } else {
                result[0] += -0.09721750874273014;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.04439210974472035;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += -0.018188428352800153;
            } else {
              result[0] += -0.12831913867478859;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.07800572824423971;
              } else {
                result[0] += -0.021003550600095136;
              }
            } else {
              result[0] += -0.08849633705283894;
            }
          } else {
            result[0] += -0.1208052675456614;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += -0.11137604244529856;
          } else {
            result[0] += -0.1404489029798089;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.142934322357179511) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
                result[0] += -0.044781861863561236;
              } else {
                result[0] += 0.04136962717761843;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)268.5000000000000568) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
                  result[0] += -0.09938463667635206;
                } else {
                  result[0] += -0.019960719110820545;
                }
              } else {
                result[0] += 0.03838514915330085;
              }
            }
          } else {
            result[0] += 0.07229427962144332;
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += 0.12329521549594413;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.04707918516777951;
                } else {
                  result[0] += -0.006284049063872858;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.03288071240136089;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)85.50000000000001421) ) ) {
                  result[0] += 0.10094405105238827;
                } else {
                  result[0] += 0.057493466743225176;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
              result[0] += 0.08085934603687861;
            } else {
              result[0] += 0.1148808149779897;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                  result[0] += -0.060831688635706364;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.056569111783655625;
                  } else {
                    result[0] += 0.007239473182941228;
                  }
                }
              } else {
                result[0] += -0.03647400803169123;
              }
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.061203844735193684;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.02304646090022805;
                    } else {
                      result[0] += 0.03828302567595282;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.14408122009792818;
                    } else {
                      result[0] += 0.0369348530830549;
                    }
                  }
                }
              } else {
                result[0] += -0.06225326503340478;
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.005944726652146043;
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.07643339020618334;
              } else {
                result[0] += 0.03501803050680216;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
              result[0] += -0.039862188352175365;
            } else {
              result[0] += 0.002012600814612932;
            }
          } else {
            result[0] += 0.018386129408873287;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
              result[0] += 0.057972696801953376;
            } else {
              result[0] += -0.07298162132890655;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += 0.06665364760172353;
            } else {
              result[0] += -0.013393604526269687;
            }
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
            result[0] += 0.1004665251888679;
          } else {
            result[0] += -0.10020128664406121;
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.014831542968751776) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += 0.08104389020652158;
            } else {
              result[0] += -0.06475134897160237;
            }
          } else {
            result[0] += -0.0036616962785328794;
          }
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.011303479029201854;
          } else {
            result[0] += -0.08530079803189185;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)33.50000000000000711) ) ) {
            result[0] += -0.04987243283894785;
          } else {
            result[0] += 0.009908255863181168;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
            result[0] += -0.1047285539549969;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.318498134613038886) ) ) {
              result[0] += -0.032801700204933774;
            } else {
              result[0] += 0.10482839590664601;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.016133129976053578;
          } else {
            result[0] += -0.05227699198502687;
          }
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.09976110731468067;
          } else {
            result[0] += -0.03649549038954152;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
          result[0] += 0.001498197350573564;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += -0.05414578473384947;
            } else {
              result[0] += -0.10927684216315298;
            }
          } else {
            result[0] += -0.12612051342866107;
          }
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.86220884323120206) ) ) {
              result[0] += -0.016452051109431983;
            } else {
              result[0] += -0.11533143229876382;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.238486170768738237) ) ) {
                result[0] += 0.013273371463652732;
              } else {
                result[0] += -0.08478897350389288;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)36.50000000000000711) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.076494216918946201) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.12144214915333192;
                  } else {
                    result[0] += 0.0946109244043256;
                  }
                } else {
                  result[0] += 0.12747560853876624;
                }
              } else {
                result[0] += -0.014014403846968863;
              }
            }
          }
        } else {
          result[0] += -0.10709867509950384;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)77.50000000000001421) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.08418746221537192;
                } else {
                  result[0] += 0.049822385752680165;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.14192860773233015;
                  } else {
                    result[0] += -0.06983213052113701;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.0486316090490492;
                    } else {
                      result[0] += 0.008805881534967305;
                    }
                  } else {
                    result[0] += -0.02416753301566084;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.028623243861435263;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.32411074638366788) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.011347720625747649;
                    } else {
                      result[0] += -0.0714834548174445;
                    }
                  } else {
                    result[0] += 0.024801317952360277;
                  }
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.05479968336676391;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)251.5000000000000284) ) ) {
                    result[0] += 0.024458071138405507;
                  } else {
                    result[0] += -0.05130566386619903;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.042255046738083446;
            } else {
              result[0] += 0.09024905141041324;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                result[0] += 0.029952202968313586;
              } else {
                result[0] += 0.07451609821416738;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.744894266128540927) ) ) {
                result[0] += -0.02469353353381885;
              } else {
                result[0] += 0.08986318735826962;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.08038958504966415;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.038113747978474034;
                } else {
                  result[0] += 0.06872769670465777;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.06874486805480533;
              } else {
                result[0] += 0.11115125798881015;
              }
            }
          }
        }
      } else {
        result[0] += -0.037680968449199465;
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.002131303138391178;
                } else {
                  result[0] += -0.06007458553596521;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.06635444427923114;
                  } else {
                    result[0] += 0.06389704828133462;
                  }
                } else {
                  result[0] += 0.009967039653026137;
                }
              }
            } else {
              result[0] += 0.06984753836345002;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
              result[0] += 0.008726309736980442;
            } else {
              result[0] += -0.05609114623417449;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.09164953290046447;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.040723489872021096;
            } else {
              result[0] += -0.06998447633973026;
            }
          }
        }
      } else {
        result[0] += -0.09195546374068647;
      }
    }
  } else {
    if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
              result[0] += 0.008593072995511602;
            } else {
              result[0] += -0.04472221900248909;
            }
          } else {
            result[0] += 0.020192719006899354;
          }
        } else {
          result[0] += -0.06987933190178029;
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.04688317860230354;
          } else {
            result[0] += -0.11355089265654704;
          }
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.03422628809604669;
            } else {
              result[0] += -0.0699308346191294;
            }
          } else {
            result[0] += 0.03700426731678446;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.276966691017151323) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.060896100009638146;
              } else {
                result[0] += 0.025265465221008594;
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += -0.0022234996582697147;
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.035699775480500906;
                } else {
                  result[0] += -0.11613452184920128;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.041173100057800394;
            } else {
              result[0] += -0.10244107941752459;
            }
          }
        } else {
          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
                result[0] += -0.03205051948681283;
              } else {
                result[0] += -0.07431584568522394;
              }
            } else {
              result[0] += -0.08058407162195397;
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.1254405351479392;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                result[0] += -0.11530530324381272;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                  if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += 0.07866271181192129;
                  } else {
                    result[0] += -0.14228702812573377;
                  }
                } else {
                  result[0] += -0.09785675368097413;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.11583107305856258;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.579273939132691318) ) ) {
                    result[0] += 0.06056554104868141;
                  } else {
                    result[0] += 0.028500654493009286;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.03140801548188078;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                      result[0] += 0.013394376734956734;
                    } else {
                      result[0] += 0.060718354697664435;
                    }
                  }
                }
              } else {
                result[0] += -0.010295540438024423;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += 0.02045612332868457;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.07048697541519658;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.856657028198243964) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += -0.08700084897936426;
                      } else {
                        result[0] += 0.015698744828129562;
                      }
                    } else {
                      result[0] += 0.06717473278470118;
                    }
                  }
                }
              } else {
                result[0] += 0.03884486615918212;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.020544858371423086;
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.999007225036621982) ) ) {
                result[0] += 0.055157474387572014;
              } else {
                result[0] += 0.10255270669792083;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.085941076278687412) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.017403194835679973;
              } else {
                result[0] += -0.0662299584412968;
              }
            } else {
              result[0] += 0.07218951895904623;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                result[0] += -0.012330707049171073;
              } else {
                result[0] += 0.0643950978934697;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                result[0] += 0.11381127957218556;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.09110317206524302;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                    result[0] += 0.09724729760453826;
                  } else {
                    result[0] += 0.013858285620151681;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.036248805606805996;
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.851041555404663974) ) ) {
                result[0] += -0.06131629045501729;
              } else {
                result[0] += 0.005302193192234243;
              }
            } else {
              result[0] += 0.015638136248488168;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)92.50000000000001421) ) ) {
                  result[0] += -0.060473195518139115;
                } else {
                  result[0] += 0.011299568180653075;
                }
              } else {
                result[0] += 0.018799091174230495;
              }
            } else {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.424057960510254794) ) ) {
                  result[0] += 0.025812559078126303;
                } else {
                  result[0] += -0.046541719298197864;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                  result[0] += -0.0009118792418377341;
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.285166740417482245) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                      result[0] += 0.06611409709749368;
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.027487472583984943;
                      } else {
                        result[0] += 0.04394004140908308;
                      }
                    }
                  } else {
                    result[0] += 0.009896236777306115;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.08751459563882155;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.034102586603732075;
            } else {
              result[0] += -0.06913851527804098;
            }
          }
        }
      } else {
        result[0] += -0.084784387864131;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)17.50000000000000355) ) ) {
            result[0] += -0.0549240016651961;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
                result[0] += 0.007690651248004849;
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.07018525471090493;
                } else {
                  result[0] += 0.007176377971613916;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.015588950213695421;
              } else {
                result[0] += -0.07259060173152589;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.01264301780473966;
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.1004926622429629;
              } else {
                result[0] += -0.05667352028862541;
              }
            } else {
              result[0] += -0.027068473369714692;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
            result[0] += 0.01779910133783207;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.665476083755494052) ) ) {
                result[0] += -0.08941301087087405;
              } else {
                result[0] += -0.014679413145692877;
              }
            } else {
              result[0] += -0.1263849149043643;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.909254074096680576) ) ) {
                result[0] += -0.03633296245853261;
              } else {
                result[0] += -0.12397022546578418;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.238486170768738237) ) ) {
                  result[0] += 0.018802455054077925;
                } else {
                  result[0] += -0.06929276811725718;
                }
              } else {
                result[0] += 0.05003672653739588;
              }
            }
          } else {
            result[0] += -0.09378254612686882;
          }
        }
      }
    } else {
      result[0] += -0.1112183848325816;
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.067782521247864214) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += 8.563322897307313e-05;
                    } else {
                      result[0] += 0.06534655616085853;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.09301272702089257;
                      } else {
                        result[0] += -0.023553470589303446;
                      }
                    } else {
                      result[0] += 0.0828011945266735;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    result[0] += -0.10094425182662632;
                  } else {
                    result[0] += 0.046795775607720955;
                  }
                }
              } else {
                result[0] += 0.02450303421447233;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.09264141231337808;
              } else {
                result[0] += -0.004435087214422483;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.0898054905657319;
              } else {
                result[0] += -0.05903403947327025;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.002607369388665197;
                  } else {
                    if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.1074612749406559;
                    } else {
                      result[0] += -0.03572038028429727;
                    }
                  }
                } else {
                  result[0] += 0.02702916718136312;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)16.50000000000000355) ) ) {
                      result[0] += -0.06674782276259367;
                    } else {
                      result[0] += 0.005563489466898892;
                    }
                  } else {
                    result[0] += 0.05161169758186841;
                  }
                } else {
                  result[0] += 0.05800187904253197;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.02241763709724985;
              } else {
                result[0] += -0.05555759837969735;
              }
            } else {
              result[0] += 0.05477882100845604;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                result[0] += 0.06481262972575876;
              } else {
                result[0] += 0.09188745881371065;
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.02685194264949153;
                } else {
                  result[0] += 0.08396674524415586;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.06630358048176974;
                } else {
                  result[0] += 0.02289545626020711;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.816582441329956943) ) ) {
            result[0] += -0.07761049606516399;
          } else {
            result[0] += 0.0024653913806361623;
          }
        } else {
          result[0] += -0.008196906889791794;
        }
      }
    } else {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
              result[0] += 0.053127423383615646;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += 0.05184836443477453;
              } else {
                result[0] += -0.09244000932795549;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                result[0] += -0.09512292634463787;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.005949356963832727;
                  } else {
                    result[0] += -0.07592893073743202;
                  }
                } else {
                  result[0] += -0.10299727523233632;
                }
              }
            } else {
              result[0] += -0.10401786865337295;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += 0.05995567022749126;
          } else {
            result[0] += -0.10511912980267782;
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.057406201083496866;
          } else {
            result[0] += -0.011950562404189256;
          }
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0032023475997499697;
          } else {
            result[0] += -0.07440848644677889;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.0019091804750273837;
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.03631395875167367;
              } else {
                result[0] += -0.08690793512716993;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.006368005568975889;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.051912069320679599) ) ) {
                    result[0] += -0.05684007576507283;
                  } else {
                    result[0] += 0.002706701942044398;
                  }
                }
              } else {
                result[0] += -0.035833574756700806;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.10242114126526312;
          } else {
            result[0] += -0.050023065958381144;
          }
        }
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += 0.03832568131197125;
          } else {
            result[0] += -0.11017262941192581;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
              result[0] += -0.05937660451076661;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
                  result[0] += 0.040836860755054195;
                } else {
                  result[0] += -0.08045523471822841;
                }
              } else {
                result[0] += -0.06804172379978346;
              }
            }
          } else {
            result[0] += -0.10711931049020273;
          }
        }
      }
    } else {
      result[0] += -0.10657767788851805;
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.045621188284381886;
            } else {
              result[0] += 0.017061212936031144;
            }
          } else {
            result[0] += -0.023703437578129698;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)63.50000000000000711) ) ) {
              result[0] += -0.10149500821565889;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06233391625101575;
              } else {
                result[0] += 0.0010891946963635809;
              }
            }
          } else {
            result[0] += 0.07199271784137966;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += 0.057970368590319454;
            } else {
              result[0] += 0.00480284427194059;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.012594390949405407;
              } else {
                result[0] += -0.13264561735246336;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += 0.019239700156818793;
                } else {
                  result[0] += 0.06978692734418795;
                }
              } else {
                if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.09091876067370401;
                } else {
                  result[0] += 0.04925316590817974;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.025667726266231913;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.050260745114673266;
                } else {
                  result[0] += -0.07732401129169941;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 9.044223807859926e-05;
              } else {
                result[0] += 0.07602034925268564;
              }
            }
          } else {
            result[0] += -0.016145563490445325;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.740175724029542792) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.07780760402957554;
          } else {
            result[0] += 0.035422874681496005;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += -0.021771468733040995;
            } else {
              result[0] += 0.0484613778219709;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.08114645774523434;
            } else {
              result[0] += -0.0822875109246441;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.09016987686777732;
            } else {
              result[0] += -0.0029713947198149685;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.03491220195810481;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0476814028026889;
                } else {
                  result[0] += -0.022595318968222596;
                }
              } else {
                result[0] += -0.011847864976113055;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += -0.10440135736588074;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.05803645811043526;
                  } else {
                    result[0] += 5.582653446076215e-05;
                  }
                } else {
                  result[0] += -0.07848621217241587;
                }
              }
            } else {
              result[0] += -0.10567618787236813;
            }
          } else {
            result[0] += -0.0950925250203346;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                    result[0] += 0.004663712905439322;
                  } else {
                    result[0] += -0.08412152965333114;
                  }
                } else {
                  result[0] += -0.09575513911035424;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.00615006428411158;
                  } else {
                    result[0] += -0.09900585152326141;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.008776971012939677;
                  } else {
                    result[0] += 0.049285699628908816;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                result[0] += 0.0006301812144904662;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.0643695286431815;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
                      result[0] += -0.04832870328479702;
                    } else {
                      result[0] += 0.011199811829811054;
                    }
                  }
                } else {
                  result[0] += -0.0836722109884852;
                }
              }
            }
          } else {
            result[0] += -0.05427228003135405;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.015624807593583824;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.033290493724445216;
              } else {
                result[0] += -0.07712664063833849;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.10058864159517014;
              } else {
                result[0] += -0.057354859359553347;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += 0.07713278381666704;
          } else {
            result[0] += -0.06683322890466957;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.001351356506349433) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
              result[0] += -0.010769922700797557;
            } else {
              result[0] += -0.11458426196690769;
            }
          } else {
            result[0] += -0.11843986017115404;
          }
        }
      }
    } else {
      result[0] += -0.10223492190107784;
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
              result[0] += 0.03198696791870354;
            } else {
              result[0] += -0.01325690137693594;
            }
          } else {
            result[0] += -0.019304696771910584;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.552201986312867099) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)59.50000000000000711) ) ) {
              result[0] += -0.10593569537342569;
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.008832812454937573;
              } else {
                result[0] += -0.05340131868890583;
              }
            }
          } else {
            result[0] += 0.054081343868251176;
          }
        }
      } else {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.13175219423462772;
                } else {
                  result[0] += -0.04667214474832043;
                }
              } else {
                result[0] += -0.03780606052881517;
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.07433852981436705;
                } else {
                  if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.044846313275554286;
                    } else {
                      result[0] += 0.006429417132236538;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01915397176203768;
                    } else {
                      result[0] += -0.04315247310179193;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
                  result[0] += -0.03924499635765985;
                } else {
                  result[0] += 0.05512360762081782;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += -0.0012537407624136636;
                  } else {
                    result[0] += 0.05495261577208016;
                  }
                } else {
                  result[0] += -0.05475171347692231;
                }
              } else {
                result[0] += -0.10802674685992125;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.603528499603273261) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                  result[0] += -0.027197404191093517;
                } else {
                  result[0] += 0.047726779819271134;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)271.5000000000000568) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.10869643035309869;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += 0.028176759516930786;
                      } else {
                        result[0] += 0.09522422792543864;
                      }
                    }
                  } else {
                    result[0] += 0.06463976940178054;
                  }
                } else {
                  result[0] += 0.04021219080639221;
                }
              }
            }
          }
        } else {
          result[0] += -0.035424667091214036;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.939840793609620917) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.06915153539296497;
              } else {
                result[0] += 0.03817684779268634;
              }
            } else {
              result[0] += -0.005231688027174308;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
              result[0] += 0.03794684321449229;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.011539425538913373;
              } else {
                result[0] += -0.08561355605534038;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.09919725634922333;
          } else {
            result[0] += 0.0452752402388821;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.07748787061235529;
              } else {
                result[0] += 0.0040698299465362206;
              }
            } else {
              if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.004399739317155156;
              } else {
                result[0] += 0.04092415567941877;
              }
            }
          } else {
            result[0] += -0.040796913974970275;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += -0.09754341400684306;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.040057668374758035;
                  } else {
                    result[0] += 0.012071972566773169;
                  }
                } else {
                  result[0] += -0.07240264565366945;
                }
              }
            } else {
              result[0] += -0.10085232652147663;
            }
          } else {
            result[0] += -0.09251375677902175;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.002370647141156101;
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                    result[0] += 0.009106527932343155;
                  } else {
                    result[0] += -0.05241849473920016;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.089241743087769443) ) ) {
                    result[0] += 0.020258015211751197;
                  } else {
                    result[0] += -0.05831216192740475;
                  }
                }
              } else {
                result[0] += -0.06246347183417966;
              }
            } else {
              result[0] += -0.011162020374411098;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.03874288288294342;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += -0.03463199243185504;
            } else {
              result[0] += -0.0816585728926222;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.001351356506349433) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
            result[0] += 0.03672185901054225;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.0025739725244726674;
            } else {
              result[0] += -0.09782798728757233;
            }
          }
        } else {
          result[0] += -0.1049657971810597;
        }
      }
    } else {
      result[0] += -0.09850079708848211;
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.198464870452881303) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.024944095199710525;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.05930271074085505;
              } else {
                result[0] += 0.002015184514123004;
              }
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)85.50000000000001421) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                    result[0] += 0.016565004842285624;
                  } else {
                    result[0] += 0.07467763093131624;
                  }
                } else {
                  result[0] += 0.03000110495926352;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)72.50000000000001421) ) ) {
                    result[0] += 0.012984711547766328;
                  } else {
                    result[0] += 0.09303129221399002;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.008569505562514313;
                    } else {
                      result[0] += 0.033039045855054926;
                    }
                  } else {
                    result[0] += 0.04718458621503519;
                  }
                }
              }
            } else {
              result[0] += -0.02320950025422558;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.015627492321257704;
              } else {
                result[0] += -0.053126426989756385;
              }
            } else {
              result[0] += 0.04648578892680201;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.06661795475638262;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += -0.05349128634366699;
                  } else {
                    result[0] += 0.026025899652781244;
                  }
                } else {
                  result[0] += 0.05481986827210298;
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.05279993367729348;
              } else {
                result[0] += 0.10017578949435768;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                result[0] += 0.05691176625148804;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                  result[0] += -0.061660781218065175;
                } else {
                  result[0] += -0.011360157713565168;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.035880791890875334;
              } else {
                result[0] += -0.03306828766273756;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                    result[0] += -0.022203878981042664;
                  } else {
                    result[0] += 0.02359063624486385;
                  }
                } else {
                  result[0] += -0.06331837623187239;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.01814719244491959;
                  } else {
                    result[0] += 0.0526585025109173;
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.11326837539672896) ) ) {
                    result[0] += 0.027315684649293584;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.007483159496103871;
                    } else {
                      result[0] += -0.0550984336052673;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.036694645701530776;
              } else {
                result[0] += 0.05712282124309386;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
            result[0] += -0.09621940031715133;
          } else {
            result[0] += 0.1566863225452319;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.0724324644200515;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.0037684579210962087;
        } else {
          result[0] += -0.06243378866965665;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.0016389651489020674;
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.05099540320206856;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)275.5000000000000568) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.033897846218662146;
                } else {
                  result[0] += -0.006025146021425898;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.004685575721144724;
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.954540252685547763) ) ) {
                    result[0] += -0.0331772748800957;
                  } else {
                    result[0] += -0.08536142435912408;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)173.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.007270934436526691;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.10747480863123181;
                } else {
                  result[0] += -0.04385316399130929;
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.658699750900269443) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.014817086426487198;
                } else {
                  result[0] += -0.05164420477782192;
                }
              } else {
                result[0] += 0.06682881814172202;
              }
            }
          } else {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.021806407391978564;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                result[0] += -0.052987651549643244;
              } else {
                result[0] += -0.08845473764022621;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
          result[0] += 0.034074405119213995;
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.12461249666839308;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += -0.11549061327942996;
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.58961367607116788) ) ) {
                result[0] += 0.04541563246802274;
              } else {
                result[0] += -0.06582480371722337;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.09510325306057515;
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
              result[0] += 0.002348335709455196;
            } else {
              result[0] += 0.039526459840707566;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.356279611587525302) ) ) {
              result[0] += -0.08831825887913199;
            } else {
              if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.04584084991533552;
              } else {
                result[0] += 0.046288623909717254;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
            result[0] += -0.07931139724067787;
          } else {
            result[0] += -0.01988361788897637;
          }
        }
      } else {
        if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += 0.06005105612789549;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.003612619638632875;
            } else {
              result[0] += 0.049525996477526105;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.032378643424071256;
            } else {
              result[0] += -0.09014721682941973;
            }
          } else {
            result[0] += 0.0908482495911588;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                result[0] += 0.045762304844710895;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += 0.07658983163984147;
                } else {
                  result[0] += -0.05340071856176618;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.03763871097649016;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.377930641174318183) ) ) {
                        result[0] += 0.018263921471059253;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.005678934024204461;
                        } else {
                          result[0] += -0.05661577991522831;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                      result[0] += 0.03184207568679655;
                    } else {
                      result[0] += -0.09198768257935558;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.03642872055510451;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += -0.09058185502961308;
                    } else {
                      result[0] += 0.025649964595302027;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                          result[0] += 0.06582773419685621;
                        } else {
                          result[0] += 0.01170131151931109;
                        }
                      } else {
                        result[0] += 0.010899214131167748;
                      }
                    } else {
                      result[0] += 0.0661427564787276;
                    }
                  } else {
                    result[0] += 0.021807803441072138;
                  }
                } else {
                  result[0] += -0.004559144270286818;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
              result[0] += 0.10747769987036207;
            } else {
              result[0] += -0.09574449440049225;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.02762021314221206;
            } else {
              result[0] += 0.02811904289564563;
            }
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07867856744228265;
            } else {
              result[0] += -0.026198075962322945;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.816582441329956943) ) ) {
              result[0] += -0.06950727461182939;
            } else {
              result[0] += 0.004260560688097438;
            }
          } else {
            result[0] += -0.009671276685331048;
          }
        } else {
          result[0] += -0.08902541963675098;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.877672910690308505) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                result[0] += -0.054667749904220325;
              } else {
                result[0] += 0.005094385687627477;
              }
            } else {
              result[0] += -0.10177266852386935;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
              result[0] += 0.0436139583389281;
            } else {
              result[0] += -0.005816984985176573;
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.051168250456332144;
            } else {
              if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.03546299452572973;
              } else {
                result[0] += 0.003419243217690172;
              }
            }
          } else {
            result[0] += -0.07020786461027313;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.04244866504420677;
          } else {
            result[0] += 0.0014957315591462381;
          }
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0878321570796023;
            } else {
              result[0] += -0.043883494967990634;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
              result[0] += -0.06445315471647582;
            } else {
              result[0] += -0.0028346889700649677;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
          result[0] += 0.03233260945247699;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.017393488900907406;
          } else {
            result[0] += -0.0793338151819275;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
              result[0] += -0.06382821665985847;
            } else {
              result[0] += 0.022148996466360787;
            }
          } else {
            result[0] += -0.09123757530824338;
          }
        } else {
          result[0] += -0.11510658147924716;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
              result[0] += 0.002874184763749635;
            } else {
              result[0] += -0.10846008806152421;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
                result[0] += 0.024515437287246266;
              } else {
                result[0] += -0.040420716292229326;
              }
            } else {
              result[0] += 0.051101910174439116;
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)276.5000000000000568) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.377930641174318183) ) ) {
              result[0] += -0.08034649112681021;
            } else {
              result[0] += -0.027412539244883023;
            }
          } else {
            result[0] += 0.02683468144666736;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
            result[0] += 0.04614009478127183;
          } else {
            result[0] += 0.0011019306673890716;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.02237129284485359;
            } else {
              result[0] += -0.0645742555571194;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
              result[0] += 0.04488577094600417;
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.08155200765377943;
              } else {
                result[0] += 0.04129330858911868;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)234.5000000000000284) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.046110805093475926;
                  } else {
                    result[0] += -0.0065389700409537845;
                  }
                } else {
                  result[0] += -0.012430061388353092;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.025582161817592444;
                } else {
                  result[0] += -0.02236131000904886;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)100.5000000000000142) ) ) {
                result[0] += -0.010680965175337053;
              } else {
                result[0] += 0.05529142950875432;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
              result[0] += 0.07606868268872655;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                  result[0] += -0.050076828781623145;
                } else {
                  result[0] += 0.05797067871489332;
                }
              } else {
                result[0] += -0.09608036942676669;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.024178597498405036;
            } else {
              result[0] += 0.02522497456152452;
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.06663102540954549;
            } else {
              result[0] += -0.015727025122457775;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.03520634327251915;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.145964622497559482) ) ) {
            result[0] += 0.012686027288316727;
          } else {
            result[0] += -0.10005964238986156;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.00550030886024066;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += -0.009611963772262127;
                } else {
                  result[0] += -0.06059418469204624;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  result[0] += 0.016909244006388165;
                } else {
                  result[0] += -0.07896422640538438;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.09830420957933214;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                  result[0] += -0.023852555121750947;
                } else {
                  result[0] += -0.08194544502241123;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.894807338714601386) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.02747551864445821;
              } else {
                result[0] += -0.09301782136891719;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.09698276911669472;
                } else {
                  result[0] += -0.03538443144453964;
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.006099456691405618;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
                      result[0] += -0.05308785119926285;
                    } else {
                      result[0] += 0.038790492912260464;
                    }
                  } else {
                    result[0] += 0.07353417805637068;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.08150768280029475) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.031304185101435664;
              } else {
                result[0] += 0.0031665882094480445;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.013890451517490214;
                  } else {
                    result[0] += -0.07783873373901513;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.033284609712136606;
                  } else {
                    result[0] += 0.030986973678546956;
                  }
                }
              } else {
                result[0] += -0.07417162210460483;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += 0.05653428886625176;
          } else {
            result[0] += -0.07014880292562058;
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.11585127468284723;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += -0.09848954058130766;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                result[0] += 0.030364584363380406;
              } else {
                result[0] += -0.09278208589428034;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.08918190258393563;
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.793003082275392401) ) ) {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)59.50000000000000711) ) ) {
              result[0] += -0.05359268296030564;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += 0.011381524973136289;
                  } else {
                    result[0] += -0.04516743814281069;
                  }
                } else {
                  result[0] += 0.028841476887469755;
                }
              } else {
                result[0] += -0.05077265761861545;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
              result[0] += 0.004366827384652711;
            } else {
              result[0] += 0.04012499222343749;
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.016182029253766963;
          } else {
            result[0] += -0.07788092532087243;
          }
        }
      } else {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.07463498593252987;
                } else {
                  result[0] += -0.004570209746053668;
                }
              } else {
                result[0] += -0.11880025155461389;
              }
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.066985936989349;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                    result[0] += -0.10793604617740112;
                  } else {
                    result[0] += 0.04927131088269629;
                  }
                }
              } else {
                result[0] += 0.014634159841308746;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
              result[0] += 0.044248931408412746;
            } else {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.030427312378350887;
                } else {
                  result[0] += -0.014444005041822947;
                }
              } else {
                result[0] += -0.025062133658841464;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += 0.055256199643310445;
            } else {
              result[0] += -0.006188251506954515;
            }
          } else {
            if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                result[0] += 0.03851317939499349;
              } else {
                result[0] += 0.0875887857155282;
              }
            } else {
              result[0] += 0.04236380393663081;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.740175724029542792) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.05888021495158451;
          } else {
            result[0] += 0.023442296477621156;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
                result[0] += 0.009271866056462913;
              } else {
                result[0] += -0.0595898544951624;
              }
            } else {
              result[0] += 0.03226704032517436;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.06545796373365106;
            } else {
              result[0] += -0.0758466132815561;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.06217826545906209;
              } else {
                result[0] += 0.0028218844780650834;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.019354218771276605;
              } else {
                result[0] += 0.02980809929568534;
              }
            }
          } else {
            result[0] += -0.055989682280618826;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
                result[0] += -0.07587761603686258;
              } else {
                result[0] += -0.02088024269088783;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.05158567423577037;
                } else {
                  result[0] += 0.04448407383648967;
                }
              } else {
                result[0] += -0.06045714533442964;
              }
            }
          } else {
            result[0] += -0.08559324938754458;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += -0.007004722534435275;
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                result[0] += -0.05551806767469676;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.0010013015764786385;
                  } else {
                    result[0] += -0.07107154189651103;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
                    result[0] += -0.003805612754245958;
                  } else {
                    result[0] += 0.03055445846313841;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.939840793609620917) ) ) {
                result[0] += -0.011193219715934738;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                  result[0] += -0.07299942488959298;
                } else {
                  result[0] += -0.017324911671954157;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.08564873878626132;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                  result[0] += -0.1114360210215978;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.05688371756122582;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.027331423610314244;
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)308.5000000000000568) ) ) {
                        result[0] += -0.07522814706196555;
                      } else {
                        result[0] += 0.10947848201003499;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0203879152667667;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
            result[0] += 0.029067311736499037;
          } else {
            result[0] += -0.06532200668763004;
          }
        } else {
          result[0] += -0.09810549622945278;
        }
      }
    } else {
      result[0] += -0.08612947197225257;
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.023808050493622515;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0016339310888269856;
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.012861377789741547;
              } else {
                result[0] += -0.07254570823685581;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)77.50000000000001421) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.06260313119799259;
                      } else {
                        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                          result[0] += -0.04038081534793152;
                        } else {
                          result[0] += 0.0375218197220437;
                        }
                      }
                    } else {
                      result[0] += 0.07149306504163806;
                    }
                  } else {
                    result[0] += 0.028175296293218313;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    result[0] += -0.10031733782522022;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.285166740417482245) ) ) {
                      result[0] += -0.01288164603826984;
                    } else {
                      result[0] += 0.04476742789321709;
                    }
                  }
                }
              } else {
                result[0] += -0.04227275798102054;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.07658845188680387;
                } else {
                  result[0] += -0.008352012414695208;
                }
              } else {
                if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.001010580138769256;
                  } else {
                    result[0] += -0.03868369454694537;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.03917640090319047;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.021257548872713458;
                    } else {
                      result[0] += -0.025570247121487324;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0478147124626004;
                  } else {
                    result[0] += -0.0038908127184974074;
                  }
                } else {
                  result[0] += 0.06749363073496542;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.06665292399655556;
                  } else {
                    result[0] += 0.0030531451725291005;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0570840748295297;
                  } else {
                    result[0] += -0.015641921752564604;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                result[0] += 0.09038314218873672;
              } else {
                result[0] += 0.05632239907403905;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.001351356506349433) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += 0.050122909234358616;
            } else {
              result[0] += -0.0582950244268049;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.05086987557959506;
            } else {
              result[0] += -0.07992290268196243;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.056093021249030095;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                  result[0] += -0.0573957573016306;
                } else {
                  result[0] += 0.008327926728374466;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.050049743172042666;
                } else {
                  result[0] += -0.03374184947575626;
                }
              }
            }
          } else {
            result[0] += -0.07387187121068842;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.06376233931193277;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.009095571196331944;
        } else {
          result[0] += -0.0550199933056274;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.997515678405763495) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.00032377303517386314;
                } else {
                  result[0] += -0.05826392672433916;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                  result[0] += -0.09588459968910662;
                } else {
                  result[0] += -0.0281028747313487;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.807895898818970615) ) ) {
                result[0] += 0.023001407098467895;
              } else {
                result[0] += -0.04023831619957055;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)45.50000000000000711) ) ) {
                  result[0] += 0.031536903588071395;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.006187075369128253;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.05407194304231841;
                    } else {
                      result[0] += 0.007167400342242987;
                    }
                  }
                }
              } else {
                result[0] += -0.052369917578150196;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.0774012749403026;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                  result[0] += -0.06969206099382;
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.021967368032861288;
                  } else {
                    result[0] += 0.07741898079806708;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.08480317773884005;
          } else {
            result[0] += -0.03919625120129563;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
          result[0] += 0.03197847979761839;
        } else {
          result[0] += -0.10435893498114567;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
        result[0] += -0.04919727443492788;
      } else {
        result[0] += -0.09604832992057272;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
              result[0] += 0.0070047883693431685;
            } else {
              result[0] += -0.0805033317602282;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
              result[0] += 1.9625235797915914e-05;
            } else {
              result[0] += 0.04752187255671716;
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)268.5000000000000568) ) ) {
            result[0] += -0.05568726789376819;
          } else {
            result[0] += 0.020393817834280903;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            result[0] += 0.04051467420228875;
          } else {
            if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.016459799490215456;
            } else {
              result[0] += 0.048816743798987546;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += -0.004666871194012506;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.06342510749433498;
              } else {
                result[0] += 0.025033072268358866;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8034253120422381) ) ) {
                  result[0] += 0.01022715665710966;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.08438373546733335;
                  } else {
                    result[0] += 0.0011438718462873944;
                  }
                }
              } else {
                if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.0953764006996134;
                } else {
                  result[0] += 0.0584832996157118;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += -0.017180782846853513;
                } else {
                  result[0] += 0.0229601689116489;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.964135169982911044) ) ) {
                  result[0] += 0.01199558291894768;
                } else {
                  result[0] += -0.06240550296463521;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.025226301169090578;
                  } else {
                    result[0] += 0.06115603234893707;
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.0020347774877028427;
                    } else {
                      result[0] += 0.10268971896468554;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                      result[0] += 0.029027754724694616;
                    } else {
                      result[0] += -0.03387266477699871;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.07751689721557353;
                } else {
                  result[0] += 0.013486758654173096;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.035794075412192546;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.04575626221146125;
                } else {
                  result[0] += 0.01334657030238575;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.894807338714601386) ) ) {
                  result[0] += 0.028529288323083324;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.0013213918368848703;
                  } else {
                    result[0] += -0.05401264273842071;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            result[0] += 0.08571988717428956;
          } else {
            result[0] += -0.08654466653762137;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.356279611587525302) ) ) {
          result[0] += 0.01352674500635885;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.023022909398818372;
          } else {
            result[0] += -0.0753414836325722;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.877672910690308505) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
                result[0] += -0.03490764892984928;
              } else {
                result[0] += 0.015395429116347104;
              }
            } else {
              result[0] += -0.05914498536070927;
            }
          } else {
            result[0] += 0.03438684028175394;
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)209.5000000000000284) ) ) {
            result[0] += -0.012380763428479923;
          } else {
            result[0] += -0.04171392025941152;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.03916448412534509;
          } else {
            result[0] += 0.0011489025458039303;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.08377791112479345;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += -0.11280388121891637;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                  result[0] += -0.04636810617946255;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += 0.1109911939495884;
                  } else {
                    result[0] += -0.017441329294187406;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.04423221653649643;
            } else {
              result[0] += 0.005105741160469426;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.807895898818970615) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
            result[0] += 0.022852982581508773;
          } else {
            result[0] += -0.07836016193210242;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
            result[0] += -0.0038888750509539086;
          } else {
            result[0] += -0.09090967945185316;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.579273939132691318) ) ) {
            result[0] += -0.08778846666845051;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
              result[0] += 0.016511349062733018;
            } else {
              result[0] += -0.072805999312876;
            }
          }
        } else {
          result[0] += -0.10838646444828096;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.439558982849121982) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.021313382283901606;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0023600598935729607;
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.015500504164780188;
              } else {
                result[0] += -0.06854129360979949;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                      result[0] += 0.010395680122280293;
                    } else {
                      if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.08770010955858039;
                        } else {
                          result[0] += 0.04287642660411154;
                        }
                      } else {
                        result[0] += 0.03385749319867762;
                      }
                    }
                  } else {
                    result[0] += 0.010070085280731594;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.06265267050062771;
                    } else {
                      result[0] += -0.03415969283776301;
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.006303082209547967;
                        } else {
                          result[0] += -0.04214717144205401;
                        }
                      } else {
                        result[0] += 0.020170073683649883;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.0513365228299395;
                        } else {
                          result[0] += -0.0040537499212331875;
                        }
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.06708262981603132;
                        } else {
                          result[0] += 0.01736975753242301;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.021758420413700824;
              }
            } else {
              result[0] += 0.048878718381250776;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.05641716220689193;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += -0.014365014215698852;
                  } else {
                    result[0] += 0.047534654373471846;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.05235642103839488;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.05695065384225814;
                  } else {
                    result[0] += -0.006409357446411832;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.07413256725334416;
                } else {
                  result[0] += 0.04019253638829272;
                }
              } else {
                result[0] += 0.0085105387193177;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.373361587524414951) ) ) {
            result[0] += 0.04159254261812243;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.042068971279995954;
            } else {
              result[0] += -0.07807363046489052;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.013862088372987203;
            } else {
              result[0] += -0.05718692894578978;
            }
          } else {
            result[0] += -0.07031113460314148;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.018543940112018816;
          } else {
            result[0] += -0.06992112614601902;
          }
        } else {
          result[0] += -0.07257165792120158;
        }
      } else {
        result[0] += 0.04646810093792339;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.07119052754060219;
          } else {
            result[0] += -0.025189430984699304;
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.011721575188909995;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                  result[0] += -0.020088050306976447;
                } else {
                  result[0] += -0.09600651703580905;
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.026738501225547873;
              } else {
                result[0] += -0.028198783429643028;
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1096682548522967) ) ) {
                result[0] += -0.0290976179563114;
              } else {
                result[0] += -0.06801189722715908;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += 0.03311771286166023;
                } else {
                  result[0] += -0.009106949817713034;
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.05007268904087347;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
                    result[0] += -0.027731214177677172;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.015319824286176007;
                    } else {
                      result[0] += 0.04797231690151553;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.570956468582154208) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.912734985351563388) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
              result[0] += 0.02810934333809444;
            } else {
              result[0] += -0.06562587956531445;
            }
          } else {
            result[0] += -0.08268369375546514;
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.10378683534555709;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.08028894549790228;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
                result[0] += -0.04149587609815758;
              } else {
                result[0] += 0.09662988505387154;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
        result[0] += -0.050775270754831264;
      } else {
        result[0] += -0.09527669269548403;
      }
    }
  }
  if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.132848501205445224) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.0018348070761732922;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.027757191591725124;
                } else {
                  result[0] += -0.07389375040112332;
                }
              } else {
                result[0] += -0.01137084366163309;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.026361960401496406;
            } else {
              result[0] += -0.0841727014012191;
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.07716661397041032;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.04550615973495345;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.03372746947362473;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
                    result[0] += -0.08508958055880786;
                  } else {
                    result[0] += 0.14495184419538099;
                  }
                }
              } else {
                result[0] += -0.06554995428110688;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
          result[0] += 0.02566203850214567;
        } else {
          result[0] += -0.09840270088319315;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
        result[0] += -0.04283438224888501;
      } else {
        result[0] += -0.09021552917970892;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.439558982849121982) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.019193766292227073;
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += -0.06176059387774725;
              } else {
                result[0] += -0.00803952215547879;
              }
            } else {
              result[0] += 0.0036928407173594624;
            }
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += 0.004958712060612773;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                      result[0] += 0.05529500743052419;
                    } else {
                      result[0] += 0.027801492472285128;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    result[0] += -0.08946501281238241;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                      result[0] += 0.0014074061087655876;
                    } else {
                      result[0] += 0.04758406094617686;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.5513958930969256) ) ) {
                  result[0] += -0.0105815534112004;
                } else {
                  result[0] += 0.030502187815667367;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.381086945533752885) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.06689362588533598;
                } else {
                  result[0] += 0.10793161183942948;
                }
              } else {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.01914377462896618;
                  } else {
                    result[0] += 0.028452782151511393;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)251.5000000000000284) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.038607034004988466;
                    } else {
                      result[0] += 0.01130679720235927;
                    }
                  } else {
                    result[0] += -0.03495383563361775;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12242221832275568) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.01869468306847614;
                    } else {
                      result[0] += -0.05333154469828003;
                    }
                  } else {
                    result[0] += 0.049367497509120624;
                  }
                } else {
                  result[0] += 0.06127120723135185;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.059322375531656146;
                  } else {
                    result[0] += 0.008918543447585782;
                  }
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.06126321626924661;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.04672467123821527;
                    } else {
                      result[0] += -0.03497984937642172;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.10904045259451892;
                } else {
                  result[0] += 0.06159125200624482;
                }
              } else {
                result[0] += 0.051101592783028615;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.05543364149600166;
              } else {
                result[0] += 0.02295001898087709;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.10136587423549526;
              } else {
                result[0] += 0.1763631100856025;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.03866404422482737;
            } else {
              result[0] += -0.07507063111505793;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.05793847834504693;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                  result[0] += -0.061985228601806566;
                } else {
                  result[0] += 0.0028729863780117735;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.03916966600750996;
                } else {
                  result[0] += -0.033674629729884824;
                }
              }
            }
          } else {
            result[0] += -0.09951787122845969;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -0.061676298066167014;
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.022320015166608767;
          } else {
            result[0] += 0.04516762924885817;
          }
        } else {
          result[0] += -0.054006167783166414;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.793003082275392401) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                result[0] += 0.0006360737177762348;
              } else {
                result[0] += 0.02376474012211774;
              }
            } else {
              result[0] += -0.017328266012468667;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                  result[0] += -0.019672748391891856;
                } else {
                  result[0] += -0.08436965481044023;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.014753899351785569;
                } else {
                  result[0] += -0.04951439469079341;
                }
              }
            } else {
              result[0] += 0.04737539842197158;
            }
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.69067406654357999) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.04336265672383775;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.022326745043368526;
                    } else {
                      result[0] += -0.04832882543872186;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.020981688741797944;
                  } else {
                    result[0] += 0.05633411079890641;
                  }
                }
              } else {
                result[0] += -0.030869468820911752;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += 0.040810845850048835;
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.02803849313476728;
                } else {
                  if ( UNLIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0035487663830850212;
                  } else {
                    result[0] += 0.03038116062391696;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.04179048486482889;
              } else {
                result[0] += -0.0035562843106523717;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.03245572149798359;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)180.5000000000000284) ) ) {
                  result[0] += 0.09294859271244071;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.07522117719623128;
                  } else {
                    result[0] += 0.033568372666401884;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.040399467525671605;
            } else {
              result[0] += 0.0006181973886951044;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.089241743087769443) ) ) {
                  result[0] += 0.016023245692173577;
                } else {
                  result[0] += -0.031097765471166744;
                }
              } else {
                result[0] += 0.04850751107283735;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                result[0] += 0.04075069619219954;
              } else {
                result[0] += -0.07091924628808653;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)234.5000000000000284) ) ) {
                  result[0] += -0.03446141297236326;
                } else {
                  result[0] += -0.08139314668318666;
                }
              } else {
                result[0] += 0.01813598432895996;
              }
            } else {
              result[0] += -0.08646201496220196;
            }
          } else {
            if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.005306476067420492;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                  result[0] += -0.03121748326756123;
                } else {
                  result[0] += 0.07292047495972244;
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.03641325677176611;
              } else {
                result[0] += 0.028221392212439494;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -0.058064701586678716;
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.009724342153681554;
        } else {
          result[0] += -0.05010401168823127;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
                result[0] += -0.027292131714430425;
              } else {
                result[0] += 0.01294948701958802;
              }
            } else {
              result[0] += -0.052162378916415876;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
              result[0] += 0.014781977533915039;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                result[0] += -0.06033733280552686;
              } else {
                result[0] += -0.0019404072677310963;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.00737050701557636;
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.08859813012433607;
              } else {
                result[0] += -0.0455924708667528;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.06703168069484454;
                } else {
                  result[0] += -0.013892100152286715;
                }
              } else {
                result[0] += -0.008772622566570984;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.90474271774292081) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
              result[0] += 0.04114057141250302;
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += -0.049261801632784076;
              } else {
                result[0] += 0.01716511412249545;
              }
            }
          } else {
            result[0] += -0.0807276264667933;
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.10186269302609258;
          } else {
            result[0] += -0.06005169252538767;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
        result[0] += -0.04709522893290371;
      } else {
        result[0] += -0.09204552910858876;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += 0.028272343845645212;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                result[0] += 0.01823257626914467;
              } else {
                result[0] += -0.04125808489344839;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.00312454354067825;
                  } else {
                    result[0] += 0.026560067087493557;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.04004906871734381;
                    } else {
                      result[0] += 0.0045909031814108416;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.040326650622325814;
                      } else {
                        result[0] += 0.04447433135569384;
                      }
                    } else {
                      result[0] += 0.04804544735960825;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
                    result[0] += -0.00792267325649396;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.03967748336858465;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.0406361374849335;
                      } else {
                        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.12389001433550786;
                        } else {
                          result[0] += -0.013269096475277074;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                    result[0] += 0.021628189674168238;
                  } else {
                    result[0] += -0.029190745155614664;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += -0.11340208906675238;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
                  result[0] += -0.04332476722560847;
                } else {
                  result[0] += 0.029068845028694473;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.019944008975281466;
                } else {
                  result[0] += 0.05632215165894158;
                }
              } else {
                result[0] += -0.009759820646942897;
              }
            } else {
              result[0] += -0.04941274844251142;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.047967923983859226;
                } else {
                  result[0] += -0.023281203946160997;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.035842394177297524;
                } else {
                  result[0] += 0.08032713779385518;
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                result[0] += 0.0474760868581503;
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.036842049918017906;
                    } else {
                      result[0] += -0.05829609863214751;
                    }
                  } else {
                    result[0] += 0.023155595461612978;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.04233590671881027;
                  } else {
                    result[0] += -0.030748359959574742;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            result[0] += 0.047200134238727404;
          } else {
            result[0] += -0.06093307394065511;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += -0.06122307210856387;
            } else {
              result[0] += 0.006748656755380303;
            }
          } else {
            result[0] += -0.12175940201359921;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.47345590591430842) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.034948021203143866;
          } else {
            result[0] += 0.00601162513195801;
          }
        } else {
          result[0] += -0.06066960914165764;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.504379272460938388) ) ) {
          result[0] += 0.009521181266134139;
        } else {
          result[0] += -0.09188817364836988;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.029520507567940182;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.722943305969239169) ) ) {
              result[0] += -0.006550801883268057;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.04663722736484772;
                } else {
                  result[0] += -0.00738506183762951;
                }
              } else {
                result[0] += -0.0954846693920246;
              }
            }
          }
        } else {
          result[0] += -0.10078445979971438;
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.509355545043946201) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)135.5000000000000284) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.00453587950343343;
              } else {
                result[0] += -0.06951225401956772;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                result[0] += -0.005000471160848498;
              } else {
                result[0] += 0.056224606379958145;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.02395861933027505;
              } else {
                result[0] += -0.05495881092186681;
              }
            } else {
              result[0] += -0.053213009070403175;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
              result[0] += -0.018001463889287982;
            } else {
              result[0] += -0.0861684819857627;
            }
          } else {
            result[0] += -0.028051317364971302;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
        result[0] += -0.04567891692867987;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.04246087840136377;
        } else {
          result[0] += -0.1034188164597028;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)290.5000000000000568) ) ) {
                result[0] += -0.03904504784024929;
              } else {
                result[0] += 0.06439149839083701;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.0013284658795545757;
              } else {
                result[0] += 0.03329755142488974;
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)209.5000000000000284) ) ) {
              result[0] += -0.004638063376370287;
            } else {
              result[0] += -0.03417734220394073;
            }
          }
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.05817722642863135;
          } else {
            result[0] += 0.0027164048472807447;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.001480159357754369;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.09843791234031073;
            } else {
              result[0] += -0.012233339921925173;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.09791205599199675;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                result[0] += -0.09581354787205229;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.053388682972339646;
                } else {
                  result[0] += -0.007412692989720824;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.03649437799716727;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.767036437988283026) ) ) {
                  result[0] += -0.10954041295037585;
                } else {
                  result[0] += 0.05454309321925648;
                }
              } else {
                result[0] += -0.02172053612543355;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.581200122833253729) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
          result[0] += 0.01980972757066083;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
            result[0] += 0.009955974123670646;
          } else {
            result[0] += -0.07231708974887477;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
              result[0] += -0.07019338708179047;
            } else {
              result[0] += 0.005441951694046815;
            }
          } else {
            result[0] += -0.07940568893237297;
          }
        } else {
          result[0] += -0.10496415013064667;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)276.5000000000000568) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                result[0] += -0.007810336757401494;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.149111986160279208) ) ) {
                  result[0] += -0.12289974048455997;
                } else {
                  result[0] += -0.019288639930961787;
                }
              }
            } else {
              result[0] += 0.027631468065975207;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.055496215820313388) ) ) {
                result[0] += 0.01549339249688915;
              } else {
                result[0] += 0.0449045918269461;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)138.5000000000000284) ) ) {
                result[0] += -0.04963175657723385;
              } else {
                result[0] += 0.031016165659936645;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)264.5000000000000568) ) ) {
            result[0] += -0.05034661396897372;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              result[0] += -0.03964254668590564;
            } else {
              result[0] += 0.03932200590652232;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
            result[0] += 0.0375027286692153;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.029552298175819704;
            } else {
              result[0] += 0.03458308017575058;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.022842565770821017;
            } else {
              result[0] += -0.05112381682026355;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.046049639305891114;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                result[0] += 0.050078180082127347;
              } else {
                result[0] += 0.08067580843446985;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                result[0] += 0.013355265717362475;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.029247861230701486;
                  } else {
                    result[0] += 0.01535686358039452;
                  }
                } else {
                  result[0] += -0.06292354126624257;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.023964864080142884;
              } else {
                result[0] += -0.02667622907902685;
              }
            }
          } else {
            if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.021266119758957674;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.03972849788289562;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)119.5000000000000142) ) ) {
                  result[0] += 0.015076129109350082;
                } else {
                  result[0] += -0.045296839073050627;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.0692161173121171;
          } else {
            result[0] += 0.1424599183715727;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.997515678405763495) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
              result[0] += 0.02525135700774568;
            } else {
              result[0] += -0.010674736153779637;
            }
          } else {
            result[0] += -0.0322955797962022;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
              result[0] += -0.00690216634237031;
            } else {
              result[0] += -0.03684396835158887;
            }
          } else {
            result[0] += -0.06921185329885458;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.145964622497559482) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.01261037942398739;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.008323715473273452;
            } else {
              result[0] += -0.049784974901000555;
            }
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += -0.03826421470524711;
                    } else {
                      result[0] += 0.033819984974417724;
                    }
                  } else {
                    result[0] += 0.052873280636215436;
                  }
                } else {
                  result[0] += 0.010827672550039157;
                }
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.05388295096344503;
                    } else {
                      result[0] += -0.026766928711192284;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.030119456790145917;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                            result[0] += 0.04702353836346215;
                          } else {
                            result[0] += 0.005695428290943845;
                          }
                        } else {
                          if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
                              result[0] += -0.05506937660999176;
                            } else {
                              result[0] += 0.014157781089187891;
                            }
                          } else {
                            result[0] += 0.02503488439824869;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.0326562855590133;
                        } else {
                          result[0] += 0.012537115999953614;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.03838411807290329;
                }
              }
            } else {
              result[0] += -0.019278404834732595;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0386889943098714;
                } else {
                  result[0] += 0.0012959984547197536;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += -0.07002773397625918;
                } else {
                  result[0] += 0.0029254043161853474;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.03520026798175865;
                } else {
                  result[0] += -0.036503401272843794;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.028824096159643094;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
                      result[0] += 0.036577816722996634;
                    } else {
                      result[0] += 0.08748222705719838;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.06463159618277109;
                    } else {
                      result[0] += 0.016048795057515216;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.971427202224732333) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
            result[0] += 0.03403180599316079;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
              result[0] += 0.04943423525274719;
            } else {
              result[0] += -0.07025735352782479;
            }
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                result[0] += -0.06328260392860685;
              } else {
                result[0] += -0.005940695642406612;
              }
            } else {
              result[0] += -0.10000384924049939;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.017722169024459593;
            } else {
              result[0] += -0.04260058424359038;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -0.05511396462515347;
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.010495318994812392;
        } else {
          result[0] += -0.047626373975539676;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.012539074608147552;
                } else {
                  result[0] += -0.02813805551057242;
                }
              } else {
                result[0] += -0.0636699523191635;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
                result[0] += 0.020011060588901095;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
                  result[0] += 0.03504899757677645;
                } else {
                  result[0] += -0.049409675868541676;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.004285414611486384;
                  } else {
                    result[0] += -0.058682099443073524;
                  }
                } else {
                  result[0] += 0.014293770370940479;
                }
              } else {
                result[0] += -0.04731335618836556;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.06721027257624358;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                    result[0] += -0.04714552540421891;
                  } else {
                    result[0] += 0.007444283489187859;
                  }
                }
              } else {
                result[0] += -0.08519432678130907;
              }
            }
          }
        } else {
          result[0] += -0.09704521799788493;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67046499252319514) ) ) {
            result[0] += -0.04357568430111072;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.024686308297107468;
            } else {
              result[0] += -0.034590843348472566;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.291543006896974433) ) ) {
            result[0] += -0.0368509919737275;
          } else {
            result[0] += -0.09912980216875895;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
        result[0] += -0.04271176366808182;
      } else {
        result[0] += -0.08772253722696725;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  result[0] += 0.0028701621370217593;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.030095061467169584;
                    } else {
                      result[0] += -0.022812760121156066;
                    }
                  } else {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.033575470377562144;
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.0345521497045676;
                      } else {
                        result[0] += 0.07859545412340695;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.006315423615330414;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.08104241589796479;
                } else {
                  result[0] += 0.01176161958716833;
                }
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.03646559743919787;
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.016865133790734396;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.001964645951528683;
                        } else {
                          result[0] += -0.03642497704673533;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.03206270530129393;
                        } else {
                          result[0] += -0.010681258748448372;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.016619504102804206;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
              result[0] += -0.10612355035680816;
            } else {
              result[0] += -0.012916724920371131;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.206374883651734287) ) ) {
            result[0] += 0.036711226343678315;
          } else {
            result[0] += -0.05403746269061669;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.02123934071616066;
          } else {
            result[0] += -0.03696939478297735;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += 0.036968963345446824;
              } else {
                result[0] += 0.08472863062660824;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.05002931972496614;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                  result[0] += 0.05485785864151237;
                } else {
                  result[0] += -0.006816550662729449;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.05678626802331369;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.03571785405068662;
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0608228111884892;
                } else {
                  result[0] += -0.03004966886452839;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        result[0] += -0.022118769921282658;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.504379272460938388) ) ) {
          result[0] += 0.008410303718629424;
        } else {
          result[0] += -0.08891198741164892;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.0007464634128743221;
                } else {
                  result[0] += -0.055293657967289606;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.03631753667521999;
                } else {
                  result[0] += -0.09453487396093967;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.384830474853516513) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                    result[0] += 0.047435583101574946;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                      result[0] += 0.0458146886692914;
                    } else {
                      result[0] += -0.007995722291005803;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                    result[0] += -0.045177116053023575;
                  } else {
                    result[0] += 0.022345417951304913;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += 0.02318321950596869;
                } else {
                  result[0] += -0.047246419237247726;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.010187420029564584;
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.0810837132192227;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.093731403350830966) ) ) {
                      result[0] += -0.0767834736964258;
                    } else {
                      result[0] += 0.09352238412278169;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.424685239791871005) ) ) {
                    result[0] += -0.037669095409022815;
                  } else {
                    result[0] += 0.001968128567766151;
                  }
                } else {
                  result[0] += -0.04301259361675895;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.07745417361680082;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06534085110031297;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.056800926760193264;
                  } else {
                    result[0] += -0.07622948176908687;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.06699494117651579;
          } else {
            result[0] += -0.02844681948836942;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
          result[0] += 0.035943313587538266;
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.11270396689542313;
          } else {
            result[0] += -0.05312275270030962;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
        result[0] += -0.03936710042988334;
      } else {
        result[0] += -0.08449298910372477;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)77.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.03465724891958207;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.01334194260975085;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                        result[0] += -0.007515477836503796;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                          result[0] += -0.008017998643072165;
                        } else {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)61.50000000000000711) ) ) {
                            result[0] += 0.032063484946222014;
                          } else {
                            result[0] += 0.05668039953623058;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                      result[0] += -0.07034492592598068;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
                        result[0] += -0.024289584131956646;
                      } else {
                        result[0] += 0.02866959319754336;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.05463803011791744;
                    } else {
                      result[0] += 0.09120336952562064;
                    }
                  } else {
                    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.00750401220982555;
                    } else {
                      result[0] += 0.04202950057768228;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.017460583302460402;
                    } else {
                      result[0] += -0.011976457458124624;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.012573981939670773;
                      } else {
                        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.07649790499541387;
                        } else {
                          result[0] += -0.002653018686479566;
                        }
                      }
                    } else {
                      result[0] += -0.05460770890164877;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.0029294747385149077;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += -0.10149997725030487;
            } else {
              result[0] += -0.012711496921587013;
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.06661952608108687;
          } else {
            result[0] += 0.11845143463334547;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.03734486955892121;
            } else {
              result[0] += 0.010464890569287887;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.0065715560694852215;
            } else {
              result[0] += -0.06559243587193281;
            }
          }
        } else {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.04022445883691364;
              } else {
                result[0] += -0.06004844378801303;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.03302498045891712;
              } else {
                result[0] += 0.06881580076039585;
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
              result[0] += 0.04056527241040278;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.03336631491658056;
              } else {
                result[0] += -0.02293241426861258;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
          result[0] += -0.03440049167667866;
        } else {
          result[0] += -0.08667893982617954;
        }
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.01907533894486754;
          } else {
            result[0] += 0.041930077821615126;
          }
        } else {
          result[0] += -0.044300990702842685;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.06416675224243723;
        } else {
          result[0] += -0.023877941015511606;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.004317399483049244;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.003780314254616876;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                      result[0] += -0.06712594428750111;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                        result[0] += -0.04368781424856691;
                      } else {
                        result[0] += 0.0662991608206388;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.08570093534194156;
                    } else {
                      result[0] += -0.02863326208193718;
                    }
                  } else {
                    result[0] += -0.0013421370760076398;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.173939466476441318) ) ) {
                  result[0] += 0.017111946095674315;
                } else {
                  result[0] += -0.05269570958380857;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += -0.08040817835734278;
                } else {
                  result[0] += -0.04126355514520639;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.025311196387830318;
            } else {
              result[0] += 0.06916767265006263;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
              result[0] += -0.006328075877254476;
            } else {
              result[0] += -0.0670378454907884;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += 0.021119949164723734;
            } else {
              result[0] += -0.09193023050403608;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
        result[0] += -0.019751527655869534;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.05604910688431906;
        } else {
          result[0] += -0.10690790018790351;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.08150768280029475) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)229.5000000000000284) ) ) {
                result[0] += -0.011565125470887217;
              } else {
                result[0] += 0.02642862306626367;
              }
            } else {
              result[0] += -0.07899994065882848;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
              result[0] += 0.009945144178964629;
            } else {
              result[0] += 0.04354435369976278;
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)254.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.04386948199163371;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                  result[0] += -0.093631739590175;
                } else {
                  result[0] += 0.018814901790464755;
                }
              }
            } else {
              result[0] += -0.029359086474380344;
            }
          } else {
            result[0] += 0.01159662652624443;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.017963224253845368;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.022448206273003472;
            } else {
              result[0] += -0.056665171853230084;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.057480547307063236;
              } else {
                result[0] += 0.019822331998005937;
              }
            } else {
              result[0] += 0.06467890707084233;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)203.5000000000000284) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.019337944381971534;
            } else {
              if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.03642141828145403;
              } else {
                result[0] += -0.002905372275541449;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                  result[0] += 0.05348715143410143;
                } else {
                  result[0] += -0.024821651794151505;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.04614834461108659;
                } else {
                  result[0] += 0.02455781680497108;
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)100.5000000000000142) ) ) {
                  result[0] += -0.03544284998710609;
                } else {
                  result[0] += 0.014754032527859395;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                    result[0] += 0.14825041626480093;
                  } else {
                    result[0] += -0.10595601783090887;
                  }
                } else {
                  result[0] += 0.011240402408584517;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.04298981098077002;
          } else {
            result[0] += -0.003852388635692507;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.740175724029542792) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.010261984283447746;
          } else {
            result[0] += 0.023000160908792863;
          }
        } else {
          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.816582441329956943) ) ) {
                result[0] += -0.043050754192615524;
              } else {
                result[0] += 0.008030384244336535;
              }
            } else {
              result[0] += -3.782749129090475e-05;
            }
          } else {
            result[0] += -0.06699893271097891;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                  result[0] += -0.0031498818896915914;
                } else {
                  result[0] += -0.07731375168798721;
                }
              } else {
                result[0] += -0.056592014989276435;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += 0.01702057199059921;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += 0.024068837983644155;
                } else {
                  result[0] += -0.041996734921341525;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.003524274465210673;
                  } else {
                    result[0] += -0.05479010496825716;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.013938845397580744;
                    } else {
                      result[0] += -0.1131173451407248;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.03973070122412545;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81531858444214045) ) ) {
                        result[0] += 0.008853584829424123;
                      } else {
                        if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.06083628596505263;
                        } else {
                          result[0] += 0.009372587731311001;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0418301221342136;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                  result[0] += -0.046197964529273106;
                } else {
                  result[0] += 0.045347667191321946;
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.04820664668079724;
                } else {
                  result[0] += -0.09012254962439742;
                }
              }
            }
          }
        } else {
          result[0] += -0.030408299152662827;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
          result[0] += 0.03945880393596554;
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.11102374369944082;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += -0.0991511684985459;
            } else {
              result[0] += -0.0034670866424893346;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.581200122833253729) ) ) {
        result[0] += -0.027043799939585268;
      } else {
        if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.02954527103609794;
        } else {
          result[0] += -0.0905247870002259;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.0019716403821771118;
        } else {
          if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.0190093116915791;
          } else {
            result[0] += -0.07474938700872533;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
                result[0] += 0.002490325647637259;
              } else {
                result[0] += 0.03623567501328778;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)268.5000000000000568) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += -0.050438191350549916;
                } else {
                  result[0] += -0.0021863517675937013;
                }
              } else {
                result[0] += 0.023052339818881943;
              }
            }
          } else {
            if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += 0.05250655563385535;
                } else {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.0363655896837553;
                  } else {
                    result[0] += -0.007257409393202662;
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.01824174759109817;
                } else {
                  result[0] += 0.03522221625085384;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.018316664332348784;
                } else {
                  result[0] += -0.059035953715655;
                }
              } else {
                result[0] += 0.06611580021836759;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
                result[0] += -0.015335987372341743;
              } else {
                result[0] += -0.05016746768866982;
              }
            } else {
              if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.011233031273690736;
                } else {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.07886229967512604;
                  } else {
                    result[0] += -0.00796570960867125;
                  }
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                    result[0] += -0.01511664258216918;
                  } else {
                    result[0] += 0.05223285422787745;
                  }
                } else {
                  result[0] += -0.04991312798067914;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.015543420558523972;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.03907707119518214;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.011205246399742096;
                        } else {
                          result[0] += -0.042794662953482436;
                        }
                      }
                    } else {
                      result[0] += 0.03970424721284071;
                    }
                  } else {
                    result[0] += 0.04602422353804625;
                  }
                } else {
                  result[0] += 0.009936520436818289;
                }
              }
            } else {
              result[0] += -0.014391741020752042;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          result[0] += 0.031394369332684545;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.039827929576281604;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
              result[0] += -0.06891113115376264;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += 0.03449081304039577;
              } else {
                result[0] += -0.08720911166787476;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.058565703967068096;
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.009407195223205057;
          } else {
            result[0] += -0.0517763882384894;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.793003082275392401) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)288.5000000000000568) ) ) {
                  result[0] += -0.025555214062605126;
                } else {
                  result[0] += 0.03842931477274103;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.006390072095449172;
                } else {
                  result[0] += -0.10359708906779613;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.579273939132691318) ) ) {
                result[0] += 0.012006360992185817;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += 0.014050424209625285;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.003906988014277714;
                  } else {
                    result[0] += -0.05020158142469511;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.008103752745894838;
                } else {
                  result[0] += -0.037569747767353376;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                    result[0] += -0.06736087574919443;
                  } else {
                    result[0] += 0.011084491076920144;
                  }
                } else {
                  result[0] += -0.030539205031861996;
                }
              }
            } else {
              result[0] += -0.06268308836924213;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
              result[0] += -0.035716590230888974;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.04362289252439469;
                } else {
                  result[0] += -0.07314248403616967;
                }
              } else {
                result[0] += -0.03045520186385677;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.291543006896974433) ) ) {
              result[0] += -0.021950982039158063;
            } else {
              result[0] += -0.09335350295284689;
            }
          }
        }
      } else {
        result[0] += -0.0933224986101551;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.581200122833253729) ) ) {
        result[0] += -0.025866593299779736;
      } else {
        result[0] += -0.07408966949546723;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
                if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += -0.02601290212096411;
                  } else {
                    result[0] += 0.0202946729163034;
                  }
                } else {
                  result[0] += 0.03431312324611656;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.01404015888855813;
                } else {
                  result[0] += -0.012604767477213364;
                }
              }
            } else {
              result[0] += -0.015817922571030176;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.69067406654357999) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                  result[0] += 0.09689921638161209;
                } else {
                  result[0] += -0.006323749809087982;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02843068994010683;
                } else {
                  result[0] += -0.010214773589501343;
                }
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.03497037309887765;
              } else {
                if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.029117321222299715;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)318.5000000000000568) ) ) {
                    result[0] += 0.004033410986624035;
                  } else {
                    result[0] += -0.03915025576456415;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            result[0] += 0.07168375167768597;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.04415814181998588;
            } else {
              result[0] += -0.1168706732662572;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.023764553645923014;
            } else {
              result[0] += -0.0039572882344740224;
            }
          } else {
            result[0] += -0.03169537002349582;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.06440901335962933;
              } else {
                result[0] += -0.02791311523728282;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.027088766082864757;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.07253446247569495;
                    } else {
                      result[0] += 0.0125950271503647;
                    }
                  } else {
                    result[0] += 0.09261357672844832;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.04777899273004297;
                    } else {
                      result[0] += -0.007117146138503477;
                    }
                  } else {
                    result[0] += 0.0849465043370605;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                result[0] += -0.036819259526707455;
              } else {
                result[0] += 0.018836330146102597;
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.053841298081094585;
              } else {
                result[0] += -0.012996777613642008;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
          result[0] += -0.005761545721445742;
        } else {
          result[0] += -0.04226476815811918;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
          result[0] += -0.006344824164050779;
        } else {
          result[0] += -0.09081977906421751;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.493027687072754794) ) ) {
            result[0] += -0.03898434243082236;
          } else {
            result[0] += -0.0799927503801823;
          }
        } else {
          result[0] += -0.020216426653269313;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                    result[0] += -0.024979469536694828;
                  } else {
                    result[0] += 0.024960261281009574;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.008478898460464786;
                  } else {
                    result[0] += -0.03204712187109202;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += -0.007336072835819623;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.08790567210212036;
                    } else {
                      result[0] += -0.029655927617851907;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.862926006317140448) ) ) {
                      result[0] += -0.06667156272333273;
                    } else {
                      result[0] += 0.04247185526242224;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.02598056506745663;
                  } else {
                    result[0] += -0.023002132555291332;
                  }
                } else {
                  result[0] += -0.058808588313300506;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += -0.07500478227740062;
                } else {
                  result[0] += -0.038067041050767636;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.06627377342507931;
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.020944448278738502;
              } else {
                result[0] += 0.056523388052606975;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
              result[0] += -0.005618940220080288;
            } else {
              result[0] += -0.06253850224893918;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += 0.018059882421803352;
            } else {
              result[0] += -0.0871734407511276;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
        result[0] += -0.03292696844927764;
      } else {
        result[0] += -0.07881426469696504;
      }
    }
  }
  if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.006169116885159191;
          } else {
            result[0] += -0.02280015984197791;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.198252916336060458) ) ) {
            result[0] += -0.01100011986191661;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)136.5000000000000284) ) ) {
                  result[0] += -0.05457853244696145;
                } else {
                  result[0] += 0.00237550097850874;
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.05499728717630175;
                } else {
                  result[0] += -0.11236742309242186;
                }
              }
            } else {
              result[0] += -0.01611651406840876;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
            result[0] += 0.05579359691789089;
          } else {
            result[0] += -0.07084768911545274;
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.10879893406367935;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += -0.09682418451987299;
            } else {
              result[0] += -0.006004341762945304;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
        result[0] += -0.03764188167555186;
      } else {
        result[0] += -0.0861100478961242;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.016581934367439828;
              } else {
                result[0] += -0.07027778608492963;
              }
            } else {
              result[0] += 0.007222357660765243;
            }
          } else {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.014906372376025469;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)128.5000000000000284) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += -0.008297694616564448;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                        result[0] += 0.055977512564845545;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                          result[0] += 0.0161973323526793;
                        } else {
                          result[0] += -0.01631817519361013;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                        result[0] += -0.008760501052067314;
                      } else {
                        result[0] += 0.03349684476309228;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                    result[0] += -0.07844300287382183;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40819787979126154) ) ) {
                      result[0] += -0.001073573589910081;
                    } else {
                      result[0] += 0.05444325220693816;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)227.5000000000000284) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += 0.016235554565977604;
                    } else {
                      result[0] += -0.055425088688383764;
                    }
                  } else {
                    result[0] += 0.026461672049687836;
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.025990527131537173;
                  } else {
                    result[0] += 0.009132428193968235;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.008202427539252221;
              } else {
                result[0] += 0.03593348690398569;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                result[0] += 0.03495125823438979;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                  result[0] += -0.07149421675474156;
                } else {
                  result[0] += 0.0011531679111157433;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                result[0] += -0.0769790780248492;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.00893990912241508;
                  } else {
                    result[0] += -0.07703170683470778;
                  }
                } else {
                  result[0] += -0.09826052201637347;
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
                result[0] += 0.015088237894394285;
              } else {
                result[0] += -0.05101627188742878;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.021231640216939796;
            } else {
              result[0] += -0.0019295933693848515;
            }
          } else {
            result[0] += -0.028328934343470547;
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.0431747800730692;
              } else {
                result[0] += -0.03795333588691896;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.03523339169757136;
              } else {
                result[0] += 0.07106493555989045;
              }
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.013668765386408453;
                } else {
                  result[0] += 0.060989879758543114;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.06022772810682719;
                } else {
                  result[0] += 0.0051342719542288905;
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.03584289843827434;
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.047612631933010276;
                } else {
                  result[0] += -0.0016822871796454517;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.05652548519693952;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
            result[0] += -0.05161916629118912;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.04757088091123463;
            } else {
              result[0] += -0.07441605211279238;
            }
          }
        } else {
          result[0] += 0.0004320302702609693;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                    result[0] += 0.008121443642098243;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.07059651697458497;
                    } else {
                      result[0] += -0.006750331666530981;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                        result[0] += -0.003812944024913804;
                      } else {
                        result[0] += 0.049235791237410675;
                      }
                    } else {
                      result[0] += 0.01709718818416458;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.07645068139172385;
                      } else {
                        result[0] += -0.0021438685546679346;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.027621290702555135;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.940638065338136542) ) ) {
                            result[0] += 0.019663271932816223;
                          } else {
                            result[0] += -0.05127162072744706;
                          }
                        }
                      } else {
                        result[0] += -0.007851939728591167;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.08150768280029475) ) ) {
                  result[0] += -0.011894617408135818;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.020119361088566212;
                  } else {
                    result[0] += -0.016424434771773324;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.0972411811498734;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.320893287658693183) ) ) {
                  result[0] += -0.037484067246201194;
                } else {
                  result[0] += 0.03003657913375306;
                }
              }
            }
          } else {
            result[0] += -0.018380430448732703;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.402451276779175693) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
              result[0] += 0.037349454701026595;
            } else {
              result[0] += -0.07097678093123146;
            }
          } else {
            result[0] += -0.05606859519320931;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.01638793410542818;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.003200660707891842;
            } else {
              result[0] += -0.051769261946126424;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.03906603205539568;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += -0.06349844260916311;
                } else {
                  result[0] += 0.013680727561095183;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.025284412078310465;
              } else {
                result[0] += 0.06457178584293469;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += 0.04495860691089928;
            } else {
              result[0] += -0.009602287661325842;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
              result[0] += -0.03580813232942062;
            } else {
              result[0] += 0.0273316403129481;
            }
          } else {
            result[0] += -0.080265512690947;
          }
        } else {
          result[0] += 0.007896763175199993;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.021594110781558266;
        } else {
          result[0] += -0.07650498807220878;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0004774184737145064;
                } else {
                  result[0] += -0.046212691786031226;
                }
              } else {
                result[0] += -0.06380499152947842;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
                result[0] += 0.011278245486092242;
              } else {
                result[0] += -0.03387288805848148;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.018823065055986743;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0501129915627763;
                      } else {
                        result[0] += 0.0069588345771703424;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.04641723632812678) ) ) {
                        result[0] += -0.04306274781700864;
                      } else {
                        result[0] += 0.054891491393128655;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.11060720579448675;
                  } else {
                    result[0] += -0.015903087366620976;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                  result[0] += -0.06417224288662549;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
                    result[0] += -0.0017989960413127002;
                  } else {
                    result[0] += -0.03534419873541403;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.07378694805618667;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.05736669676950584;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.049683320454014;
                  } else {
                    result[0] += -0.07439015183941734;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
            result[0] += 0.012771856793243051;
          } else {
            result[0] += -0.03149816505858306;
          }
        }
      } else {
        result[0] += -0.08647101188753215;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.581200122833253729) ) ) {
        result[0] += -0.020613508800498476;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.03827218517333992;
        } else {
          result[0] += -0.09356398829528355;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.04780562510206325;
          } else {
            result[0] += -0.016261094272689253;
          }
        } else {
          result[0] += -0.009644581588513368;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
          result[0] += 0.05076468114830132;
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.10571066994174257;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += -0.08532500803890802;
            } else {
              result[0] += -0.0007177889466299375;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.402451276779175693) ) ) {
        result[0] += -0.010822390353686281;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.043416331099975745;
        } else {
          result[0] += -0.10077907173555006;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
              result[0] += 0.0053182958279109965;
            } else {
              result[0] += -0.029681609768899116;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.089241743087769443) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.02824942723134192;
                  } else {
                    result[0] += 0.05708020612532168;
                  }
                } else {
                  result[0] += 0.04129589167464043;
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.11216060237743533;
                    } else {
                      result[0] += 0.02091228232559774;
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.400584220886231357) ) ) {
                      result[0] += 0.017327558307766874;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.03031325207969142;
                        } else {
                          result[0] += -0.035607523698030365;
                        }
                      } else {
                        result[0] += -0.038702629791262884;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.004061003895277808;
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                      result[0] += 0.04335077927455894;
                    } else {
                      result[0] += -0.03908647175682474;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0042430749217876405;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.08302568606319688;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                          result[0] += -0.07037924330542876;
                        } else {
                          result[0] += -0.004114967281525482;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.011553435846182831;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)32.50000000000000711) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.590043783187867099) ) ) {
                        result[0] += 0.024601353529235374;
                      } else {
                        result[0] += 0.05673553564449516;
                      }
                    } else {
                      result[0] += 0.01321177642951326;
                    }
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += -0.054200393393506;
                      } else {
                        result[0] += 0.0001909597238395383;
                      }
                    } else {
                      result[0] += 0.022328574338635977;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    result[0] += 0.017296667973389454;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                        result[0] += 0.0029207039878700754;
                      } else {
                        result[0] += -0.07216194099608117;
                      }
                    } else {
                      result[0] += 0.03278277347696263;
                    }
                  }
                } else {
                  result[0] += -0.04153644701401045;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += -0.10793167027797024;
          } else {
            result[0] += 0.1376236283477754;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                result[0] += -0.03174244112428767;
              } else {
                result[0] += 0.012893884103932555;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.010241975330169142;
              } else {
                result[0] += 0.06268796859008267;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += -0.060567126156051726;
            } else {
              result[0] += -0.008325810518734822;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                  result[0] += 0.00401613802443615;
                } else {
                  result[0] += 0.058543576614237675;
                }
              } else {
                result[0] += -0.015388712983527498;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.033011157598562525;
              } else {
                result[0] += 0.06773019066786523;
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.192109584808350498) ) ) {
                  result[0] += -0.005319803604408847;
                } else {
                  result[0] += 0.06318529776899101;
                }
              } else {
                result[0] += 0.04667820831888289;
              }
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.03214981301221588;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.03942059428768712;
                } else {
                  result[0] += -0.03030932443037417;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.939840793609620917) ) ) {
          result[0] += -0.0062522705756019835;
        } else {
          result[0] += -0.06073186461717947;
        }
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.029346423597710933;
        } else {
          result[0] += 0.017713679558001058;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)77.50000000000001421) ) ) {
                result[0] += 0.016350753982966072;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.293085813522339311) ) ) {
                  result[0] += 0.05079902660971117;
                } else {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.009138631750552656;
                  } else {
                    result[0] += 0.012107116699998097;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                result[0] += -0.06324413432775182;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.664231777191163886) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)55.50000000000000711) ) ) {
                    result[0] += -0.04556569079664252;
                  } else {
                    result[0] += 0.006807415052268968;
                  }
                } else {
                  result[0] += 0.016151133392874863;
                }
              }
            }
          } else {
            result[0] += -0.030260307904841507;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.402451276779175693) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.006850502791803244;
            } else {
              result[0] += -0.032526822838912964;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)300.5000000000000568) ) ) {
                    result[0] += -0.026319912407472862;
                  } else {
                    result[0] += 0.10247815530817944;
                  }
                } else {
                  result[0] += 0.03692206066843656;
                }
              } else {
                result[0] += 0.05092604999414985;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0011097293377715582;
                  } else {
                    result[0] += -0.051426431029350274;
                  }
                } else {
                  result[0] += 0.046420593465402865;
                }
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                    result[0] += -0.0005065229063184743;
                  } else {
                    result[0] += 0.060444696871503584;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.026468284759515875;
                  } else {
                    result[0] += -0.02628317308086797;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            result[0] += 0.036997243365917414;
          } else {
            result[0] += -0.058958448303205915;
          }
        } else {
          result[0] += -0.04938303456012715;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.01739636521528498;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
            result[0] += 0.008465309646076538;
          } else {
            result[0] += -0.08592252546827987;
          }
        }
      } else {
        result[0] += 0.04844649929715436;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0031709095323874275;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.003884172945335679;
                } else {
                  result[0] += -0.10379989386732205;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                result[0] += 0.03463075295837076;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += 0.015720738431941;
                } else {
                  result[0] += -0.020774465751200524;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.056037901549526425;
            } else {
              result[0] += -0.0011774192304683663;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.008074229794727028;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)33.50000000000000711) ) ) {
                      result[0] += -0.12116440319416272;
                    } else {
                      result[0] += -0.01560285650750721;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.09448420209682473;
                      } else {
                        result[0] += 0.030140998242877845;
                      }
                    } else {
                      result[0] += 0.02998089672229247;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.091865496082257;
                    } else {
                      result[0] += -0.028291048573653905;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                      result[0] += -0.04756677855614397;
                    } else {
                      result[0] += 0.04719400975227318;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += -0.025989446156707882;
                } else {
                  result[0] += 0.060917184368633776;
                }
              } else {
                result[0] += -0.037325569909668024;
              }
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.09087287268151646;
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)36.50000000000000711) ) ) {
                result[0] += 0.03138341080566999;
              } else {
                result[0] += -0.06935196315209262;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.740175724029542792) ) ) {
              result[0] += -0.031269369124400107;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.02360017210981313;
              } else {
                result[0] += -0.02752425059166705;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.07823445480447194;
              } else {
                result[0] += -0.02163102283592535;
              }
            } else {
              result[0] += -0.01047631606346551;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.377930641174318183) ) ) {
            result[0] += -0.018260055849691738;
          } else {
            result[0] += -0.09229634617233778;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.968900680541993964) ) ) {
        result[0] += -0.025528262909833063;
      } else {
        result[0] += -0.0707375872855785;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)197.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.02853906540599978;
                  } else {
                    result[0] += -0.00843415816397359;
                  }
                } else {
                  result[0] += 0.02514855528651996;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.012025263905952462;
                } else {
                  result[0] += -0.012010107474154642;
                }
              }
            } else {
              result[0] += -0.011942295446388364;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                  result[0] += 0.0897536893635261;
                } else {
                  result[0] += 0.002469783303979006;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.029039464536158428;
                } else {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.054217682496996905;
                  } else {
                    result[0] += 0.011717187632578627;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0038939835936309345;
                } else {
                  result[0] += -0.05198265654504666;
                }
              } else {
                if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.02303270117876173;
                } else {
                  result[0] += 0.002232813799353529;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.06804662223536287;
          } else {
            result[0] += -0.06238354846521699;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.011588603800255247;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.01366067166615183;
                } else {
                  result[0] += -0.04504616791527999;
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.011328668557359242;
              } else {
                result[0] += 0.05441107008688787;
              }
            }
          } else {
            result[0] += -0.018938589673396386;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.03372473648143181;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.025416532614105293;
                } else {
                  result[0] += 0.025600886209064356;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                result[0] += 0.075840215471078;
              } else {
                result[0] += 0.03969713930312832;
              }
            }
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.026569189254896115;
              } else {
                result[0] += -0.031779143512379336;
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.04614747385062504;
              } else {
                result[0] += -0.013233067735591825;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
          result[0] += -0.011955542097476549;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.023009012426841426;
          } else {
            result[0] += -0.0733015509528837;
          }
        }
      } else {
        result[0] += 0.04571918681990828;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.00562127114075346;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.002927783085902849;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.09419393641106462;
                    } else {
                      result[0] += -0.02212828870499299;
                    }
                  }
                }
              } else {
                result[0] += -0.05925022476963421;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.877672910690308505) ) ) {
                result[0] += 0.013962321110619586;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += 0.01851576785546746;
                } else {
                  result[0] += -0.032648417935768115;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.015121544445925795;
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
                      result[0] += -0.005167165312866801;
                    } else {
                      result[0] += -0.08233682380238144;
                    }
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.349482774734497958) ) ) {
                      result[0] += 0.03006600457792585;
                    } else {
                      if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += -0.04665479200096135;
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
                          result[0] += 0.03990811851441458;
                        } else {
                          result[0] += -0.17711222545384042;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)166.5000000000000284) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.424685239791871005) ) ) {
                    result[0] += -0.034571433689500046;
                  } else {
                    if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02512162908153423;
                    } else {
                      result[0] += 0.014688033064395107;
                    }
                  }
                } else {
                  result[0] += -0.03819753120426502;
                }
              }
            } else {
              result[0] += -0.07303141752667618;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
              result[0] += -0.0026045970957188323;
            } else {
              result[0] += 0.09944517299486352;
            }
          } else {
            result[0] += -0.027587116300577242;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
          result[0] += 0.05041631696290643;
        } else {
          result[0] += -0.09829090340285002;
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
        result[0] += -0.030317569794889088;
      } else {
        result[0] += -0.07958053605348224;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.375737190246582919) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.020918914609097533;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.350240230560303178) ) ) {
                  result[0] += -0.043025418013512146;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += -0.022371045888209276;
                  } else {
                    result[0] += 0.022256380877108437;
                  }
                }
              }
            } else {
              result[0] += -0.1058195787144629;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += 0.01102863464249315;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.08564307874334545;
              } else {
                result[0] += -0.03275371880270375;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
              result[0] += 0.05519847180526874;
            } else {
              result[0] += -0.029086788899567808;
            }
          } else {
            result[0] += -0.011470315916340311;
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.009811690350107012;
            } else {
              result[0] += -0.02532968485451255;
            }
          } else {
            result[0] += 0.029637442115951124;
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += 0.028618457885081536;
            } else {
              result[0] += -0.012353738822663694;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.043121202226168255;
              } else {
                result[0] += -0.010879440040259418;
              }
            } else {
              result[0] += 0.051724124059372045;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                    result[0] += -0.07186912894473353;
                  } else {
                    result[0] += 0.0060628018998925975;
                  }
                } else {
                  result[0] += -0.04210306418090046;
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += 0.019107256860315003;
                } else {
                  result[0] += -0.10890351176686258;
                }
              }
            } else {
              if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.005226433352892097;
              } else {
                result[0] += 0.03448079873861554;
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.060384412579186246;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.860674262046814409) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.306375980377199042) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                      result[0] += 0.03411716119224201;
                    } else {
                      result[0] += 0.002363079464303903;
                    }
                  } else {
                    result[0] += -0.03142469824932858;
                  }
                } else {
                  result[0] += -0.025754203509833064;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.001047464382781093;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.016547161643058677;
                  } else {
                    result[0] += 0.05323866705258034;
                  }
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                      result[0] += 0.01756107155620547;
                    } else {
                      result[0] += 0.03959021706918195;
                    }
                  } else {
                    result[0] += -0.010556611452615385;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.03921224550138431;
          } else {
            if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.017005579571015787;
            } else {
              result[0] += -0.03535582090894021;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
            result[0] += -0.042637598048708404;
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.006212301971464671;
            } else {
              result[0] += -0.07035077384039953;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.020723147894732775;
          } else {
            result[0] += -0.030388458373348033;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.597218394279480425) ) ) {
            result[0] += -0.02909294225283861;
          } else {
            result[0] += -0.06960520807446449;
          }
        } else {
          result[0] += -0.016111038404371045;
        }
      } else {
        if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
              result[0] += 0.001718529754779621;
            } else {
              result[0] += 0.036615346701289755;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.579273939132691318) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.009136183919371385;
              } else {
                result[0] += -0.022586451935802103;
              }
            } else {
              result[0] += -0.061651251297510735;
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.388237953186036044) ) ) {
                result[0] += 0.009958743046309122;
              } else {
                result[0] += -0.017813189732221627;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.030140984911050824;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += -0.09165789413496477;
                } else {
                  result[0] += 0.020269425292164193;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              result[0] += -0.012879589577808055;
            } else {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += -0.036689855478884585;
              } else {
                result[0] += -0.08248169853685983;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.527194023132325107) ) ) {
        result[0] += -0.03293191761586066;
      } else {
        result[0] += -0.09225068467788197;
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.0003744828659645888;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
              result[0] += 0.010318474965452068;
            } else {
              result[0] += -0.07057082296569214;
            }
          } else {
            result[0] += -0.0011732435362363206;
          }
        }
      } else {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)77.50000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                    result[0] += -0.009168920725548684;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0973596322638376;
                    } else {
                      result[0] += -0.035989199710297924;
                    }
                  }
                } else {
                  result[0] += 0.03445561167818897;
                }
              } else {
                result[0] += 0.02039700707398874;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.04975318085801918;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.26837396621704279) ) ) {
                      result[0] += -0.05207143117634884;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.02472215219496437;
                      } else {
                        result[0] += -0.1198538835209724;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.0513365447479792;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.026734785181233553;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                        result[0] += -0.09181355471543862;
                      } else {
                        result[0] += 0.0247571231483388;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0022351225611953515;
                } else {
                  result[0] += 0.037058393726098586;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.064940579862305;
              } else {
                result[0] += -0.011805142165688837;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.018382329967196593;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.939840793609620917) ) ) {
                  result[0] += 0.01694426087579488;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.04529508726879534;
                  } else {
                    result[0] += -0.018434040713156016;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += -0.0023677552082140207;
                } else {
                  result[0] += 0.037537185159227805;
                }
              } else {
                result[0] += 0.043292407852099576;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.02891103621023207;
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.032310371502228735;
                } else {
                  result[0] += -0.011597114477055782;
                }
              }
            }
          } else {
            result[0] += 0.03953156508243042;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          result[0] += 0.028425408090515122;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
            result[0] += 0.02597624771177412;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
              result[0] += -0.06747375288502828;
            } else {
              result[0] += 0.004208354418322118;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.02073090508663659;
          } else {
            result[0] += -0.06287135295838864;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.013427926678859654;
            } else {
              result[0] += 0.04685930052927647;
            }
          } else {
            result[0] += -0.055411667435584344;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.013617640309324814;
            } else {
              result[0] += 0.033581414429385746;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.0005592515693587802;
            } else {
              result[0] += -0.05243263245750677;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.356279611587525302) ) ) {
              result[0] += -0.06777426602764466;
            } else {
              result[0] += -0.012054408222897765;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.03741639550938466;
            } else {
              result[0] += 0.016608804731327403;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.008860181665485052;
            } else {
              result[0] += -0.014028816654569065;
            }
          } else {
            result[0] += -0.02584130108567624;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)164.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                result[0] += -0.04936149332402794;
              } else {
                result[0] += 0.008359828757661263;
              }
            } else {
              result[0] += -0.07087419874095403;
            }
          } else {
            result[0] += -0.09299868463799998;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
          result[0] += -0.030295103480876814;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.04013140997524678;
            } else {
              result[0] += -0.06507899327942576;
            }
          } else {
            result[0] += -0.026180464105067098;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.291543006896974433) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += 0.0034929433880356298;
          } else {
            result[0] += -0.07794903264226269;
          }
        } else {
          result[0] += -0.09011909195239573;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
                result[0] += 0.00010021126721863563;
              } else {
                result[0] += -0.05643780059719229;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                result[0] += -0.013363075084108373;
              } else {
                result[0] += -0.06737197781348998;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.02908294790501751;
            } else {
              result[0] += -0.0011988091245536923;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.549732685089113104) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)59.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.015391828884923917;
                  } else {
                    result[0] += -0.05636948408248026;
                  }
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.02115092795154123;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
                      result[0] += -0.02723909651288824;
                    } else {
                      result[0] += 0.00396130016624776;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)302.5000000000000568) ) ) {
                  result[0] += 0.025614181906826485;
                } else {
                  result[0] += -0.01420618074355736;
                }
              }
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.4455442428588885) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.016241349238678485;
                      } else {
                        result[0] += -0.029958420800155566;
                      }
                    } else {
                      result[0] += 0.03565666773301353;
                    }
                  } else {
                    if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.02054880613942643;
                      } else {
                        result[0] += 0.03587802873993296;
                      }
                    } else {
                      result[0] += 0.058605849907024846;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.013498777829859572;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                        result[0] += -0.022217265791810808;
                      } else {
                        result[0] += 0.04868279615671308;
                      }
                    } else {
                      result[0] += 0.03930466398094212;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.027401467929505235;
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.04454427598684507;
                  } else {
                    result[0] += 0.004909818431055665;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                result[0] += 0.03865947000475792;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += -0.044305291937423455;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)199.5000000000000284) ) ) {
                    if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.040644133911052455;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                        result[0] += 0.018929333399168025;
                      } else {
                        result[0] += 0.05593128221241799;
                      }
                    }
                  } else {
                    result[0] += -0.011827212567808831;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                  result[0] += -0.05560883798288296;
                } else {
                  result[0] += 0.0014331486639986245;
                }
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.03469846173165557;
                  } else {
                    result[0] += -0.05411501343537672;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.06860942423021606;
                    } else {
                      result[0] += -0.006499282611162807;
                    }
                  } else {
                    result[0] += 0.030575725077394207;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.402451276779175693) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            result[0] += 0.031303073817421;
          } else {
            result[0] += -0.0577270225547359;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            result[0] += -0.05310813921079345;
          } else {
            result[0] += 0.06255816830273188;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
          result[0] += -0.02448276267353662;
        } else {
          result[0] += -0.08050120169822339;
        }
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.015242680705024;
        } else {
          result[0] += -0.03418565843140264;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.089241743087769443) ) ) {
            result[0] += -0.0057676033013555035;
          } else {
            result[0] += -0.056850959699907624;
          }
        } else {
          result[0] += -0.015773611914122635;
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)65.50000000000001421) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)33.50000000000000711) ) ) {
            result[0] += -0.010932938412849503;
          } else {
            result[0] += 0.011878713478434073;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.0023581585937398238;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  result[0] += 0.012597626180419812;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += -0.01102169615635205;
                  } else {
                    result[0] += -0.06728612171024716;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += -0.013045640852804748;
              } else {
                result[0] += -0.048297941482231044;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
              result[0] += 0.022478025780326545;
            } else {
              result[0] += -0.09568414765082427;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.402451276779175693) ) ) {
        result[0] += -0.0073948909181447186;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.037332962146777275;
        } else {
          result[0] += -0.09652683116252825;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)89.50000000000001421) ) ) {
              result[0] += 0.016653886422264588;
            } else {
              if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += -0.0017287639428921226;
                } else {
                  result[0] += -0.03138782267164971;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += 0.0291782366027461;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.026954205525156386;
                  } else {
                    result[0] += 0.010340073494266399;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
              result[0] += -0.05590287120350643;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.36324071884155451) ) ) {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.004656539947820156;
                } else {
                  result[0] += -0.024862025365888444;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.053055324009614914;
                } else {
                  result[0] += -0.02664436262713915;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.00029871710464163584;
            } else {
              result[0] += 0.02581940423316382;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += 0.014989756764482349;
              } else {
                result[0] += 0.07683363139408705;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.02909327381723863;
              } else {
                result[0] += 0.0016907845170156368;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09344673156738459) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.11321719853734552;
          } else {
            result[0] += -0.01238873263625386;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += 0.046923302619425095;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
                result[0] += 0.005230851764011918;
              } else {
                result[0] += 0.03336613363613236;
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.024601526077339675;
            } else {
              result[0] += 0.04783132813153783;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.04654029524304785;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
            result[0] += -0.01488104747301077;
          } else {
            result[0] += -0.07434234726920724;
          }
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.025572688843209308;
          } else {
            result[0] += -0.024901323767924297;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.0013591107012876592;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.0599890756900506;
            } else {
              result[0] += 0.07631105520659767;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
            result[0] += -0.0645279621880761;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.549732685089113104) ) ) {
              result[0] += -0.02824225116544504;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.08010730976105854;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.046964118628787056;
                } else {
                  result[0] += 0.04236424916929788;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
              result[0] += 0.003209587455378007;
            } else {
              result[0] += -0.042398658996884886;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.031527491177387124;
              } else {
                result[0] += -0.08985830354543435;
              }
            } else {
              result[0] += -0.09581340144937169;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                  result[0] += -0.06349136733179932;
                } else {
                  result[0] += -0.006056371777186777;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
                  result[0] += 0.026996986922288038;
                } else {
                  result[0] += -0.042602301989160735;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)65.50000000000001421) ) ) {
                  result[0] += 0.007528084842237154;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.010787153572982135;
                  } else {
                    result[0] += -0.051303731610874995;
                  }
                }
              } else {
                result[0] += -0.07687376975876256;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.07447421668868734;
              } else {
                result[0] += -0.007397839703416229;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                result[0] += -0.030661743030196364;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                    result[0] += 0.006726535392697537;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.051889732475744424;
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.03367106572275876;
                      } else {
                        result[0] += 0.062249011072744324;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.10060120391851873;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                      result[0] += -0.058985856193393194;
                    } else {
                      result[0] += 0.08847517647619765;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.968900680541993964) ) ) {
        result[0] += -0.0209302332129266;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.018144265335834808;
        } else {
          result[0] += -0.082033781816914;
        }
      }
    }
  }
}

