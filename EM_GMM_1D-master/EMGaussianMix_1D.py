import numpy as np
import matplotlib.pyplot as plt   
import scipy.stats as ss
import imageio.v2 as imageio

def randomGen(mean,SD,pis,N):
    #This function generated GMM samples based on proved means, SD's, and Pi's
    #array of samples to be returned
    gen = np.zeros(N)
    #random array to select from which gaussian the smaple is taken
    sourcedist = np.random.uniform(0,1,N)
    for x in range(N):
        #these ifs choose which gaussian to take sample from based on Pis
        if sourcedist[x] < pis[0]:
            gen[x] = np.random.normal(mean[0],SD[0])  
        elif sourcedist[x] < pis[0] + pis[1]:
            gen[x] = np.random.normal(mean[1],SD[1])
        else:
            gen[x] = np.random.normal(mean[2],SD[2])
    return gen


def main():
    #Parameters
    N = 500
    mean = [1,5,9]
    SD = [1,1,1]
    pis = [.6,.15,.25]
    #Data Generation
    data = randomGen(mean,SD,pis,N) 
    #Plotting Parameters:
    plt.ion()
    fig = plt.figure(figsize = (12,8)) 
    ax = fig.add_subplot(111)
    x = np.linspace(-2,12,100)
    ytrue = pis[0]*ss.norm(mean[0],SD[0]).pdf(x)+\
            pis[1]*ss.norm(mean[1],SD[1]).pdf(x)+\
            pis[2]*ss.norm(mean[2],SD[2]).pdf(x)
    filenames = []              #gif files storage
    #Initialization of EM parameters:
    muK = np.array([0,4,11])
    varK = np.array([1.5,1.5,1.5])
    PiK = np.array([.5,.2,.3])
    GammaNK = np.zeros((3,N))
    Nk = np.zeros(3)
    den = None

    #Just dummy old values for while loop convergence
    oldmuK = muK - 1
    oldvarK = varK - 1
    oldPiK = PiK - 1
    eps = .01 #Arbitrary, mostly just meant so that convergence is reasonably quick
    iteration = 0
    while(1):
        #Do plotting and gif creation things
        y = PiK[0]*ss.norm(muK[0],np.sqrt(varK[0])).pdf(x)+\
            PiK[1]*ss.norm(muK[1],np.sqrt(varK[1])).pdf(x)+\
            PiK[2]*ss.norm(muK[2],np.sqrt(varK[2])).pdf(x)
        ax.plot(x,y,'b-', label ='EM Distribution')
        ax.plot(x,ytrue,'g-',label = 'True Distribution')
        ax.hist(data,20,density = True,color = 'orange',label = 'Random Data')
        ax.legend(loc = 'best', frameon = False)
        fig.canvas.draw()       #update figure
        filetitle = "ExpMax_1D_" + str(iteration) + ".png"   
        fig.savefig(filetitle)
        filenames.append(filetitle)       #Append to list of file names for gif
        iteration += 1

        #Check if Converged
        print("Mu", muK,"Var", varK, "Pi", PiK)
        if(np.linalg.norm(muK-oldmuK)+np.linalg.norm(varK-oldvarK)+np.linalg.norm(PiK-oldPiK) < eps):
            break
        ax.clear()  #clears axes for next iteration
        #Store Old Values
        oldmuK = muK
        oldvarK = varK
        oldPiK = PiK
        Nk = np.zeros(3) #reset Nk for calculating it (usually M-step, but more efficient in E)
        #Calculate Gamma(Z_NK) for the E step
        for k in range(3):
            for n in range(N):
                den = 0
                for j in range(3):
                    den += PiK[j]*ss.norm(muK[j],np.sqrt(varK[j])).pdf(data[n])  #Note the sqrt bc this func takes SD not var
                GammaNK[k,n] = PiK[k]*ss.norm(muK[k],np.sqrt(varK[k])).pdf(data[n])/den
                #Calculate NK while I'm at it
                Nk[k] += GammaNK[k,n] #Application of Eq 9.27
        #Complete M-step here with analytical equations
        PiK = Nk/N  #Eq 9.26
        muK = np.zeros(3) #reset muK
        varK = np.zeros(3) #reset varK 
        #Eq 9.24;
        for k in range(3):
            for n in range(N):
                muK[k] += GammaNK[k,n]*data[n]
        muK = np.divide(muK,Nk)
        #Eq 9.25
        for k in range(3):
            for n in range(N):
                varK[k] += GammaNK[k,n]*(data[n]-muK[k])**2
        varK = np.divide(varK,Nk)
        #Ended Loop

    #Compile Images into gif and save (comment out if you dont want gif)
    with imageio.get_writer('ExpMax_1D.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

if __name__ == "__main__":
    main()