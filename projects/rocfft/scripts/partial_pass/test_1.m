function val = test_1(tpt, wgs, lengths, batch, factors_pp)

lengths_reord = [lengths(1) lengths(3) lengths(2)];
tpb = wgs / tpt;

wgs_scaled = wgs * max(factors_pp);

no_blocks = ceil(lengths_reord(2) / tpb);
no_blocks = no_blocks * batch * lengths_reord(3) / max(factors_pp)

val = (prod(lengths) * batch) / (no_blocks * wgs_scaled * factors_pp(1) );

