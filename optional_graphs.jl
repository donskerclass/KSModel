# this file contains optional graphs which we may or may not want to include
# it should be run after running KS.jl


unc = ell.^(-1/γ).<=wgrid # unconstained people
uncsim  = unc'*diagm(wweights)*μsim
consim  = (1-unc)'*diagm(wweights)*μsim

figure()
ax=axes()
ax[:tick_params]("both",labelsize=18)
plot(1:dur,consim')
xlabel("t")