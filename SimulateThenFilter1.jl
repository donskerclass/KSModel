#SimulateThenFilter1.jl
#Exactly as SimulateThenFilter.jl but updated to syntax for plotting to current Plots.jl syntax
#Simulate draws from one version of Huggett model, then Kalman filter to see performance of state and shock recovery

##If not done, run the model to get dynamics before running this
#include("Huggett.jl")

using Random
rng=MersenneTwister(5678)



SimT = 200 #Periods to simulate

nx = size(hx,1)
ny = size(gx,1)

obsdim    = 5 #R, Mean of c & w, variance of c & w
shockdim  = 2 #Productivity, risk are two shocks in this version of model

obscov  = 0.1 * I(obsdim)

shockcorr = 0.3;
shockstd  = 0.4;

shockcov  = shockstd*[1 shockcorr; shockcorr 1]

shocknoise=shockcov*randn(rng,shockdim,SimT)
obsnoise=obscov*randn(rng,obsdim,SimT)

simX=zeros(nx,SimT)

shocklocs=zeros(nw+1,shockdim) #Matrix placing shocks in appropriate location in x state
shocklocs[nw,1]=1
shocklocs[nw+1,2]=1
simX[:,1]=shocklocs*shocknoise[:,1] #Initialize with shock around steady state
for t=2:SimT
  simX[:,t]=hx*simX[:,t-1]+shocklocs*shocknoise[:,t]
end

simY=gx*simX

#Simulation in terms of grid values
simxp = C2Vx*Qx'simX
simyp = C2Vy*Qy'simY

simell   = simyp[1:nw,:]
simr   = simyp[nw+1,:]
simm   = simxp[1:nw,:]
simag  = simxp[nw+1,:]
simsig = simxp[nw+2,:]

simc = dcdxss*[simell;simr';simag'];

#Plots

wgrid = repeat(wg,1,SimT);
tgrid = repeat((1:SimT)',maxw,1)
f1 = plot(tgrid,wgrid,simm[1:maxw,:],st = :surface,xlabel="t",ylabel="Cash",title="Simulated Cash Distribution Deviations",legend=false)
f2 = plot(tgrid,wgrid,simc[1:maxw,:],st = :surface,xlabel="t",ylabel="Cash",zlabel="Consumption",title="Simulated Consumption Function Deviations",legend=false)
f3 = plot(simr,xlabel="t",ylabel="R",title="Simulated Interest Rate Deviations",legend=false)
f4 = plot(simag,xlabel="t",ylabel="Z",title="Simulated Aggregate Income Shock Deviations",legend=false)
f5 = plot(simsig,xlabel="t",ylabel="σ",title="Simulated Risk Shock Deviations",legend=false)
savefig(f1, "plots/Huggett/HuggettSimulatedm.png")
savefig(f2, "plots/Huggett/HuggettSimulatedc.png")
savefig(f3, "plots/Huggett/HuggettSimulatedr.png")
savefig(f4, "plots/Huggett/HuggettSimulatedag.png")
savefig(f5, "plots/Huggett/HuggettSimulatedls.png")

#Observable series

simmeanc=dmcdxss'*[simc;simm]+obsnoise[1,:]'
simvarc=dvcdxss'*[simc;simm]+obsnoise[2,:]'
simmeanw=dmwdmss'*simm+obsnoise[3,:]'
simvarw=dvwdmss'*simm+obsnoise[4,:]'
simrobs=simr'+obsnoise[5,:]'

obseries=[simmeanc;simvarc;simmeanw;simvarw;simrobs]


Hc=dcdxss*[C2Vy*Qy'*gx;shocklocs[:,1]'] #map from x space to consumption (on grid points)
Hmtemp=C2Vx*Qx'
Hm=Hmtemp[1:nw,:] #map from x space to m(w) (on grid points)
Hrtemp=C2Vy*Qy'*gx
Hr=Hrtemp[nw+1,:] #map from x space to R


obsmat=[dmcdxss'*[Hc;Hm];
      dvcdxss'*[Hc;Hm];
      dmwdmss'*Hm;
      dvwdmss'Hm;
      Hr[:]']

obseries2=obsmat*simX+obsnoise   # Should be the same as obseries, if code is right: checks say yes


#Initialize with 0 initial condition
xhat0=zeros(nx)
shockcovbig=shocklocs*shockcov*shocklocs'
sighat0=shockcovbig


## Kalman Filter!!!
using QuantEcon

kfilt=Kalman(hx,obsmat,shockcovbig,obscov)
set_state!(kfilt, xhat0, sighat0)

kalmeans=zeros(nx,SimT)
kalvars=zeros(nx,nx,SimT)


for t=1:SimT
  kalmeans[:,t]=kfilt.cur_x_hat
  kalvars[:,:,t]=kfilt.cur_sigma
  update!(kfilt, obseries[:,t])
end

kalpts=C2Vx*Qx'*kalmeans
kalm=kalpts[1:nw,:]
kalag=kalpts[nw+1,:]
kalsig=kalpts[nw+2,:]

sigTvar=zeros(SimT,1)
agTvar=zeros(SimT,1)
for t=1:SimT
  sigTvar[t]=kalvars[nx,nx,t] #Predicted variance of latent sigma shock
  agTvar[t]=kalvars[nx-1,nx-1,t] #Predicted variance of latent ag shock
end

tgrid=1:SimT
f6 = plot(tgrid,kalsig,label="Predicted Shock",xlabel="t",title="True and Predicted Mean Risk Shock")
plot!(tgrid,simsig,label="True Shock")
savefig(f6, "plots/Huggett/HuggettKfiltls.png")


f7 = plot(tgrid,kalag,label="Predicted Shock",xlabel="t",title="True and Predicted Mean Income Shock")
plot!(tgrid,simag,label="True Shock")
savefig(f7, "plots/Huggett/HuggettKfiltag.png")


thingtoplot = kalm[1:maxw,:]
xgrid = repeat(wg,1,SimT);
ygrid = repeat((1:SimT)',maxw,1);
f8 = plot(xgrid,ygrid,thingtoplot,st = :surface,xlabel="Cash",ylabel="t",
  title="Predicted Deviation in Cash Distribution",legend=false)
savefig(f8, "plots/Huggett/HuggettKfiltm.png")



## USE FRBNY Code for Kalman Filtering and Smoothing
# Currently not functional!

#  using StateSpaceRoutines

#  CCC=zeros(nx)
#  DD=zeros(obsdim)

# ##Here is kalman filter code:
# log_likelihood, z, P, pred, vpred, kfilt, kvfilt, yprederror, ystdprederror, rmse, rmsd, z0, P0 = kalman_filter(obseries, hx, shocklocs, CCC, shockcov, obsmat, DD, obscov, xhat0, sighat0; allout=true, n_presample_periods=0)

# # Compare: QuantEcon version gives predicted state x_t+1|t, comparable to "pred"
# agdif=kalag-vec(pred[nw,:])
# sigdif=kalsig=vec(pred[nw+1,:])
# # Result: they differ at start, then converge: maybe this code uses steady state filter? Or initial conditions had an issue, not sure

# #plot (components of) kfilt against true (unobservable) series

# figure(48)
# plot(tgrid,kfilt[nw,:],tgrid,simag)
# legend(("Filtered Shock","True Shock"))
# title("True and Filtered Productivity Shock")
# xlabel("t")

# figure(49)
# plot(tgrid,kfilt[nw+1,:],tgrid,simsig)
# legend(("Filtered Shock","True Shock"))
# title("True and Filtered Risk Shock")
# xlabel("t")

# ## Smoothing: try different routines

# smoothed_states, smoothed_shocks=hamilton_smoother(obseries, hx, shocklocs, CCC, shockcov, obsmat, DD, obscov, xhat0, sighat0; n_presample_periods=0)

# figure(49)
# plot(tgrid,smoothed_states[nw+1,:],tgrid,kfilt[nw+1,:],tgrid,simsig)
# legend(("Smoothed Shock","Filtered Shock", "True Shock"))
# title("True, Filtered, and Smoothed Risk Shock")
# xlabel("t")

# figure(50)
# plot(tgrid,smoothed_states[nw,:],tgrid,kfilt[nw,:],tgrid,simag)
# legend(("Smoothed Shock","Filtered Shock", "True Shock"))
# title("True, Filtered, and Smoothed Aggregate Shock")
# xlabel("t")
