
#include "header.h"

void predict_unit1(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.700598716735840066) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.7512402534484881) ) ) {
          result[0] += 0.07701941640825774;
        } else {
          result[0] += 0.14694403722274485;
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.11511120972368534;
        } else {
          result[0] += 0.04219284898326181;
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        result[0] += 0.17968790508962573;
      } else {
        result[0] += 0.09914312449386971;
      }
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.04818582489971618;
          } else {
            result[0] += -0.0722532888874569;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.16988942486445213;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.206118345260621005) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.019654653402883325;
              } else {
                result[0] += -0.15068505878252808;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += 0.20003180195365466;
              } else {
                result[0] += -0.15885477760638358;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.055836200714113104) ) ) {
            result[0] += -0.06613566699892452;
          } else {
            result[0] += -0.14650271204408813;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += 0.019953931624919014;
                  } else {
                    result[0] += -0.12045952245334023;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.13589342578349836;
                  } else {
                    result[0] += 0.08884054148572584;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.081657409667969194) ) ) {
                  result[0] += -0.03221753109888999;
                } else {
                  result[0] += -0.1580059809680716;
                }
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.029279103594561165;
              } else {
                result[0] += 0.08954474562994619;
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.19752502441406428) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.007651964398590411;
                  } else {
                    result[0] += -0.06700550454160627;
                  }
                } else {
                  result[0] += 0.07894861534691432;
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.081657409667969194) ) ) {
                    result[0] += -0.06274378079607025;
                  } else {
                    result[0] += -0.15402363830710025;
                  }
                } else {
                  result[0] += -0.04336453023384222;
                }
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.028034348238061044;
                } else {
                  result[0] += -0.12733838017778407;
                }
              } else {
                result[0] += 0.08449972335858835;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.1445712488346173;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0253822010488168;
              } else {
                result[0] += -0.18246366298067318;
              }
            } else {
              result[0] += -0.14014604993665322;
            }
          }
        } else {
          result[0] += 0.12715494626675475;
        }
      } else {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0224081170542182;
            } else {
              result[0] += -0.16468669204246222;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.1822080612182635) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.599987030029298651) ) ) {
                    result[0] += -0.036549819036376156;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.02295267753934166;
                    } else {
                      result[0] += 0.13699751184756795;
                    }
                  }
                } else {
                  result[0] += 0.11699361660043595;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                      result[0] += -0.009347657100076954;
                    } else {
                      result[0] += -0.09981064811879181;
                    }
                  } else {
                    result[0] += -0.008650222747658063;
                  }
                } else {
                  result[0] += 0.09597562870520925;
                }
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.651049375534058505) ) ) {
                  result[0] += 0.12123284294841526;
                } else {
                  result[0] += 0.009570317339563136;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += -0.0716487829554774;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.11180592756145835;
                    } else {
                      result[0] += 0.008522941812973117;
                    }
                  } else {
                    result[0] += 0.13324268095734082;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += 0.12096625585996884;
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.20289884052725082;
              } else {
                result[0] += -0.012661800416406396;
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.05771746437277339;
              } else {
                result[0] += -0.12379571258087058;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.060242575852383545;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.1416564274205269;
                    } else {
                      result[0] += -0.03823043371363973;
                    }
                  } else {
                    result[0] += 0.14502658826526918;
                  }
                } else {
                  result[0] += 0.1790980844262203;
                }
              }
            }
          }
        }
      }
    }
  }
}

