# #Imports
import functools
import operator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from time import sleep
from tqdm import tqdm
import time
from numba import jit,cuda
from threading import Thread
from multiprocessing import Process
import multiprocessing
from joblib import Parallel,delayed
import pandas as pd
import base64
import random
import threading


def img2chromo(img_arr):
    chromosome = np.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))
    return chromosome

def chrom2img(chromosome, img_shape):
    img_arr = np.reshape(a=chromosome, newshape=img_shape)
    return img_arr


# User Adjustments
POPULATION_SIZE = [200]##number of Chromosomes
MaxGen = 1500
Pc = .9995 #porbability of crossover, the left over .1 is chance at mutation
El = .1 #Percentage Elite
MutAd = .005 #when our fitness has less then a .5% change from the previous best fitness, increase our mutation
FM = 1 #Which fitness method to use, 1:mean, 2: number difference, 3:non correct pixels
imgCreate = 1500 #create new image every X generations
image = Image.open("IMG1.jpg") #select Image to be the target
THR=10 #number of threads

image_array = np.array(image)
IMG_Arr2 = img2chromo(image_array)
Gene = np.unique(IMG_Arr2)


# Valid genes
#GENES = range(0,256)
def GENES(cols=0):
    # if cols ==0:
    #     color = np.random.choice(range(256),size=1)
    #     return color
    # else:
    #     color = list(np.random.choice(range(256),size=(cols)))
    #     return color
    if cols ==0:
        color = np.random.choice(Gene,size=1)
        return color
    else:
        color = list(np.random.choice(Gene,size=(cols)))
        return color

# Target string to be generated
TARGET = IMG_Arr2
subTarget = None

class Individual(object):
    '''
    Class representing individual in population
    '''

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()
        self.pixscore = self.cal_pixscore()
        self.pixclosesc = self.pixclosesc()

    @classmethod
    def mutated_genes(self):
        '''
        create random genes for mutation
        '''
        global GENES
        #gene = random.choice(GENES())
        gene = GENES()
        return gene

    @classmethod
    def create_gnome(self):
        '''
        create chromosome or string of genes
        '''
        gnome_len = GENES(subTarget.shape)
        return gnome_len


    def mate(self, Par2T):
        '''
        Perform mating and produce new offspring
        '''

        # chromosome for offspring
        MutR = 1-Pc
        par1 = self.chromosome
        par2 = Par2T.chromosome
        choice = np.random.randint(2,size=len(self.chromosome)).astype(bool)
        newchoice = np.where(choice,par1,par2)
        mutateval = self.mutated_genes()
        MutChoice = np.random.choice([0,1],size=len(self.chromosome),p=((Pc),(MutR)))
        finalchild = np.where(MutChoice>0,mutateval,newchoice)
        count = np.count_nonzero(MutChoice > 0)

        #finalchild = np.where(randarr>=MutR,mutateval,newchoice)
        child_chromosome = finalchild
        return Individual(child_chromosome)

    def cal_fitness(self):
        '''
        Calculate fitness score, it is the number of
        characters in string which differ from target
        string.
        '''
        global TARGET
        fitness = 0
        if FM==1:
            #Method1 - get the mean of the pixel difference
            count = np.mean(np.abs(subTarget-self.chromosome))
            fitness+=count
        elif FM==2:
            #Method2 - get the absolute difference
            #res = np.abs(self.chromosome - TARGET)
            res = np.abs(subTarget - self.chromosome)
            count = np.sum(res)
            fitness+=count
        elif FM==3:
            #Method3 - only increase fitness if pixel not correct
            count = np.count_nonzero(np.abs(subTarget-self.chromosome))
            fitness += count
            fitness = np.int64(fitness)

        return fitness

    def cal_pixscore(self):
        incorrect = np.count_nonzero(np.abs(subTarget-self.chromosome))
        pixscore = len(subTarget)-incorrect
        return pixscore

    def pixclosesc(self):
        res2 = np.abs(subTarget - self.chromosome)
        difcount = int(np.sum(res2))
        tarcount = int(np.sum(subTarget))
        countres = tarcount-difcount
        if countres<=0:countres=0
        res = (countres/tarcount)*100
        res = round(res, 5)
        return res


def MainRunCode(size,result,index):
    global POPULATION_SIZE
    global subTarget
    spoint=index*size
    epoint = spoint+size
    if index == (THR-1):
        subTarget=IMG_Arr2[spoint:]
    else:
        subTarget = IMG_Arr2[spoint:epoint]
    lastfit = 1
    global Pc
    def_Pc = Pc
    MutSpikeGenWait=1000
    LastMutSpike=0

    initalpop = []

    bestfit=0

    # plt.ion()
    # fig = plt.figure()
    # ax1=fig.add_subplot(111)

    # create initial population
    print(f'\nThread {index} Start \n')
    for popsize in POPULATION_SIZE:
        printpoint = popsize/10
        pipcount=0
        population = []
        for _ in range(popsize):
            gnome = Individual.create_gnome()
            population.append(Individual(gnome))
            if pipcount==printpoint:
                # print("THR:"+str(index)+ " "+str(_) +'/'+str(popsize))
                pipcount=0
            pipcount+=1
        initalpop.append(population)
    totfit=[]
    totpix=[]
    topchromo=[]
    FinGen = []  # final Generation
    toppixerr=[]
    runtime=[]

    for fm in range(1,4):
        substart=time.time()
        global FM
        FM=fm
        corpixnum=[]
        fitscrOT=[]
        bchromo = []  # save best chromosome
        fingen = []
        corpixerr = []
        for population in initalpop:
            newpop = []
            for peep in population:
                newpop.append(Individual(peep.chromosome))
            population=newpop
            MGRun = tqdm(range(MaxGen))
            fitscrOTTMP=[]
            corpixnumTMP=[]
            corpixnumerrTMP=[]
            # current generation
            generation = 0
            for _ in MGRun:
                # sort the population in increasing order of fitness score
                population = sorted(population, key=lambda x: x.fitness)
                sleep(.01)
                MGRun.set_description("STP: "+str(fm) +" THR:"+str(index)+" Pop: %s" % str(len(population)) +" Fitness: %s" % str(population[0].fitness)+" PixelID: %s" % str(population[0].pixclosesc))

                # if the individual having lowest fitness score ie.
                # 0 then we know that we have reached to the target
                # and break the loop
                fitscrOTTMP.append(population[0].fitness)
                corpixnumTMP.append(population[0].pixscore)
                corpixnumerrTMP.append(population[0].pixclosesc)

                if population[0].fitness <= 0:
                    break

                # Otherwise generate new offsprings for new generation
                new_generation = []
                fitness = [o.fitness for o in population]
                # From 50% of fittest population, Individuals
                halfpop = int(len(population)/2)
                norm_fit_vals = fitness[:halfpop]/sum(fitness[:halfpop])
                cumsum = np.zeros(len(norm_fit_vals))

                for i in range(len(norm_fit_vals)):
                    valadd = np.sum(norm_fit_vals[i:])
                    cumsum[i]=cumsum[i]+valadd
                # Perform Elitism, 10% of fittest population
                # goes to the next generation
                Elite = El*100
                Child = 100-Elite
                s = int((Elite * len(population)) / 100)
                new_generation.extend(population[:s])

                # mate to produce offspring
                s = int((Child * len(population)) / 100)

                for _ in range(s): #create list of parents
                    R = np.random.rand()
                    if R < cumsum[-1]: R = cumsum[-1]
                    p1Index, = np.where(cumsum <= R)
                    parent1 = population[p1Index[0]]
                    # parent1 = np.random.choice(population[:halfpop],p=cumsum[:halfpop])
                    R = np.random.rand()
                    if R < cumsum[-1]: R = cumsum[-1]
                    p2Index, = np.where(cumsum <= R)
                    parent2 = population[p2Index[0]]
                    # parent2 = np.random.choice(population[:halfpop],p=cumsum[:halfpop])
                    child = parent1.mate(parent2)
                    new_generation.append(child)

                population = new_generation
                Pc = def_Pc

                # process = [multiprocessing.Process(target=matebaby,args=(p1,p2,output))for x in range(len(p1))]
                # for p in process:
                #     p.start()
                # for p in process:
                #     p.join()
                # results = [output.get() for p in process]



                # # for _ in range(s):
                #
                #     #parent2 = np.random.choice(population[:halfpop],p=cumsum[:halfpop])
                #     child = parent1.mate(parent2)
                #     new_generation.append(child)

                population = new_generation
                Pc=def_Pc

                if generation % 25 == 0 : #old display updates
                    diff = 1 - (population[0].fitness / lastfit)
                    lastfit = population[0].fitness
                    if diff<MutAd and generation!=0:
                        if generation>LastMutSpike and (generation-LastMutSpike)>=MutSpikeGenWait:
                            LastMutSpike=generation
                            Pc = Pc-0.1
                    # print("Generation: " + str(generation) + " Fitness: " + str(population[0].fitness) + " PopSize: " + str(POPULATION_SIZE)+" Diff: "+str(diff)+" Mut: "+str(1-Pc))

                if generation %imgCreate ==0 and bestfit!=population[0].fitness and generation!=0 : #image generation every
                    f, ax = plt.subplots(1, 2)
                    #ax[0].imshow(TARGET,interpolation='nearest')
                    global image_array
                    disTARGET = chrom2img(TARGET,image_array.shape)
                    ax[0].imshow(disTARGET)
                    ax[0].set_title('TARGET')
                    #ax[1].imshow(population[0].chromosome,interpolation='nearest')
                    disChrom = chrom2img(population[0].chromosome, image_array.shape)
                    ax[1].imshow(disChrom)
                    ax[1].set_title('Best Chromosome')
                    plt.suptitle("Pop:"+str(len(population))+" Generation: "+str(generation)+" Fit: "+str(population[0].fitness)+" Correct Pixels: "+str(population[0].pixscore))
                    plt.show()
                    bestfit = population[0].fitness
                generation += 1

            corpixnum.append(corpixnumTMP)
            corpixerr.append(corpixnumerrTMP)
            fitscrOT.append(fitscrOTTMP)
            bchromo.append(population[0])
            fingen.append(generation)
        topchromo.append(bchromo)
        totfit.append(fitscrOT)
        totpix.append(corpixnum)
        toppixerr.append(corpixerr)
        FinGen.append(fingen)
        subeltime = time.time() - substart
        runtime.append(subeltime)
    result[index]=[topchromo,totfit,totpix,FinGen,toppixerr,runtime]
    print(f'\nThread {index} Stop\n')

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)

# # Driver code
def main():
    st = time.time()
    subpros = int(len(IMG_Arr2) / THR)
    manager = multiprocessing.Manager()
    returnDic = manager.dict()
    jobs=[]
    for i in range(THR):
        #if i == (THR-1):
            #subtmp = len(IMG_Arr2)-(subpros*THR)
            #subpros=subpros+subtmp
        p = multiprocessing.Process(target=MainRunCode,args=(subpros,returnDic,i))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    newdickeys = returnDic.keys()
    newdicvals = returnDic.values()
    tmptopchromo = []
    tmptotfit=[]
    tmptotpix=[]
    tmpFinGen=[]
    tmptotpixErr=[]
    tmptimerun=[]
    for indx in range(len(newdickeys)):
        index = newdickeys.index(indx)
        vals = newdicvals[index]
        tmptopchromo.append(vals[0])
        tmptotfit.append(vals[1])
        tmptotpix.append(vals[2])
        tmpFinGen.append(vals[3])
        tmptotpixErr.append(vals[4])
        tmptimerun.append(vals[5])

    topchromo=None
    toppix=None
    topfit=None
    toppixper=None

    for i in range(len(POPULATION_SIZE)):
        hold=[]
        holdpx=[]
        holdfit=[]
        holdpxper=[]
        count=0
        for item in tmptopchromo:
            count+=1
            if hold:#not empty
                hold[0][0] = np.append(hold[0][0],item[0][i].chromosome)
                hold[1][0]=np.append(hold[1][0],item[1][i].chromosome)
                hold[2][0]=np.append(hold[2][0],item[2][i].chromosome)
                holdpx[0][0] = np.append(holdpx[0][0],item[0][i].pixscore)
                holdpx[1][0]=np.append(holdpx[1][0],item[1][i].pixscore)
                holdpx[2][0]=np.append(holdpx[2][0],item[2][i].pixscore)
                holdfit[0][0] = np.append(holdfit[0][0],item[0][i].fitness)
                holdfit[1][0]=np.append(holdfit[1][0],item[1][i].fitness)
                holdfit[2][0]=np.append(holdfit[2][0],item[2][i].fitness)
                holdpxper[0][0] = np.append(holdpxper[0][0],item[0][i].pixclosesc)
                holdpxper[1][0]=np.append(holdpxper[1][0],item[1][i].pixclosesc)
                holdpxper[2][0]=np.append(holdpxper[2][0],item[2][i].pixclosesc)
            else:#is empty
                hold=[[item[0][i].chromosome],[item[1][i].chromosome],[item[2][i].chromosome]]
                holdpx=[[item[0][i].pixscore],[item[1][i].pixscore],[item[2][i].pixscore]]
                holdfit=[[item[0][i].fitness],[item[1][i].fitness],[item[2][i].fitness]]
                holdpxper=[[item[0][i].pixclosesc],[item[1][i].pixclosesc],[item[2][i].pixclosesc]]
        holdpx[0]=sum(holdpx[0][0])
        holdpx[1] = sum(holdpx[1][0])
        holdpx[2] = sum(holdpx[2][0])
        holdfit[0]= (sum(holdfit[0][0])/count)
        holdfit[1] = (sum(holdfit[1][0])/count)
        holdfit[2] = (sum(holdfit[2][0])/count)
        holdpxper[0]=(sum(holdpxper[0][0]/count))
        holdpxper[1]=(sum(holdpxper[1][0]/count))
        holdpxper[2]=(sum(holdpxper[2][0]/count))
        if topchromo:
            topchromo.append([[hold[0]],[hold[1]],[hold[2]]])
            toppix.append([[holdpx[0]],[holdpx[1]],[holdpx[2]]])
            topfit.append([[holdfit[0]],[holdfit[1]],[holdfit[2]]])
            toppixper.append([[holdpxper[0]],[holdpxper[1]],[holdpxper[2]]])
        else:
            topchromo=[[hold[0]],[hold[1]],[hold[2]]]
            toppix=[[holdpx[0]],[holdpx[1]],[holdpx[2]]]
            topfit=[[holdfit[0]],[holdfit[1]],[holdfit[2]]]
            toppixper=[[holdpxper[0]],[holdpxper[1]],[holdpxper[2]]]

    FinGen=None
    for i in range(len(POPULATION_SIZE)):
        hold=[]
        count=0
        for item in tmpFinGen:
            count+=1
            if hold:#not empty
                hold = [(hold[0]+item[0][i]), (hold[1]+item[1][i]), (hold[2]+item[2][i])]
            else: #is empty
                hold=[item[0][i],item[1][i],item[2][i]]
        hold[0] = int(hold[0]/count)
        hold[1] = int(hold[1]/count)
        hold[2] = int(hold[2]/count)
        if FinGen:
            FinGen.append([[hold[0]],[hold[1]],[hold[2]]])
        else:
            FinGen = [[hold[0]],[hold[1]],[hold[2]]]

    hold=[]

    totfit=None
    for i in range(len(POPULATION_SIZE)):
        hold=[]
        count=0
        for item in tmptotfit:
            count+=1
            if hold:#not empty
                hold = [(hold[0]+[item[0][i]]), (hold[1]+[item[1][i]]), (hold[2]+[item[2][i]])]
            else: #is empty
                hold=[[item[0][i]],[item[1][i]],[item[2][i]]]
        if totfit:
            totfit.append(hold)
        else:
            totfit = hold

    hold=[]

    totpix=None
    for i in range(len(POPULATION_SIZE)):
        hold=[]
        count=0
        for item in tmptotpix:
            count+=1
            if hold:#not empty
                hold = [(hold[0]+[item[0][i]]), (hold[1]+[item[1][i]]), (hold[2]+[item[2][i]])]
            else: #is empty
                hold=[[item[0][i]],[item[1][i]],[item[2][i]]]
        if totpix:
            totpix.append(hold)
        else:
            totpix = hold

    hold=[]

    totpixerr=None
    for i in range(len(POPULATION_SIZE)):
        hold=[]
        count=0
        for item in tmptotpixErr:
            count+=1
            if hold:#not empty
                hold = [(hold[0]+[item[0][i]]), (hold[1]+[item[1][i]]), (hold[2]+[item[2][i]])]
            else: #is empty
                hold=[[item[0][i]],[item[1][i]],[item[2][i]]]
        if totpixerr:
            totpixerr.append(hold)
        else:
            totpixerr = hold

    finruntime=None
    for i in range(len(POPULATION_SIZE)):
        hold=[]
        count=0
        for item in tmptimerun:
            count+=1
            if hold:#not empty
                hold = [(hold[0]+item[0]), (hold[1]+item[1]), (hold[2]+item[2])]
            else: #is empty
                hold=[item[0],item[1],item[2]]
        hold[0] = int(hold[0]/count)
        hold[1] = int(hold[1]/count)
        hold[2] = int(hold[2]/count)
        if finruntime:
            finruntime.append([[hold[0]],[hold[1]],[hold[2]]])
        else:
            finruntime = [[hold[0]],[hold[1]],[hold[2]]]

    #Graph of fitness over generations
    for p in range(len(POPULATION_SIZE)):#Threads Fitness per Gen Plot
        fig,axs = plt.subplots(len(totfit),figsize=(15,15))
        for j in range(len(totfit)):
            fit = totfit[j]
            #plt.figure()
            for i in range(len(fit)):
                axs[j].plot(range(100,len(fit[i])),fit[i][100:],label='THR:'+str(i)+' Pop:'+str(POPULATION_SIZE[p]))
            # axs[j].xlabel("Generations")
            # axs[j].ylabel("Fitness Score")
            axs[j].set_title("Fitness over Generation for Fitness Model "+str(j+1))
            axs[j].legend(loc="upper right")

        for ax in axs.flat:
            ax.set(xlabel='Generations',ylabel='Fitness Score')
        for ax in axs.flat:
            ax.label_outer()


    for p in range(len(POPULATION_SIZE)): #Threads Correct Pixel Per Gen Plots
        fig, axs = plt.subplots(len(totfit), figsize=(15, 15))
        for j in range(len(totpix)):
            pix = totpix[j]
            #plt.figure()
            for i in range(len(pix)):
                axs[j].plot(range(len(pix[i])),pix[i],label='THR:'+str(i)+' Pop:'+str(POPULATION_SIZE[p]))
            # plt.xlabel("Generations")
            # plt.ylabel("Correct Pixel Count")
            axs[j].set_title("Correct Pixels over Generation for Fitness Model "+str(j+1))
            axs[j].legend(loc="lower right")
        for ax in axs.flat:
            ax.set(xlabel='Generations', ylabel='Correct Pixel Count')
        for ax in axs.flat:
            ax.label_outer()

    for p in range(len(POPULATION_SIZE)):#AVG Correct Pixel per Gen Plots
        fig, axs = plt.subplots(len(totpix)+1, figsize=(15, 15), sharex=True,sharey=True)
        for j in range(len(totpix)):
            #plt.figure()
            pix = totpix[j]
            y , error = tolerant_mean(pix)
            axs[len(totpix)].plot(np.arange(len(y))+1,y,label='Model:'+str(j+1) +' AVG for Pop: '+str(POPULATION_SIZE[p]))
            axs[j].plot(np.arange(len(y))+1,y,color='green',label='Model:'+str(j+1) +' AVG for Pop: '+str(POPULATION_SIZE[p]))
            axs[j].fill_between(np.arange(len(y))+1,y-error,y+error,color='lime',label='Model:'+str(j+1) +' Error')
            # plt.xlabel("Generations")
            # plt.ylabel("Correct Pixel Count")
            axs[j].set_title("AVG Correct Pixels over Generations Model "+str(j+1))
            axs[j].legend(loc="lower right")
            #plt.legend()
        axs[len(totpix)].legend(loc="lower right")
        axs[len(totpix)].set_title("All Models AVG Correct Pixels Over Generations")
        for ax in axs.flat:
            ax.set(xlabel='Generations', ylabel='Correct Pixel Count')
        for ax in axs.flat:
            ax.label_outer()

    for p in range(len(POPULATION_SIZE)):
        fig, axs = plt.subplots(len(totpixerr) + 1, figsize=(15, 15), sharex=True, sharey=True)
        for j in range(len(totpixerr)):
            pix = totpixerr[j]
            y, error = tolerant_mean(pix)
            axs[len(totpixerr)].plot(np.arange(len(y)) + 1, y,
                     label='Model:' + str(j + 1) + ' AVG for Pop: ' + str(POPULATION_SIZE[p]))
            axs[j].plot(np.arange(len(y)) + 1, y, color='green',
                     label='Model:' + str(j + 1) + ' AVG for Pop: ' + str(POPULATION_SIZE[p]))
            axs[j].fill_between(np.arange(len(y)) + 1, y - error, y + error, color='lime',
                             label='Model:' + str(j + 1) + ' Error')
            # plt.xlabel("Generations")
            # plt.ylabel("Percent Identical")
            axs[j].set_title("AVG Percentage Identical over Generations Model "+str(j+1))
            axs[j].legend(loc="lower right")
        axs[len(totpixerr)].legend(loc="lower right")
        axs[len(totpixerr)].set_title("All Models AVG Percentage Identical over Generation")
        for ax in axs.flat:
            ax.set(xlabel='Generations', ylabel='Percent Identical')
        for ax in axs.flat:
            ax.label_outer()

    eltime = time.time() - st
    extime = time.strftime("%H:%M:%S", time.gmtime(eltime))
    timstr = str('  Total Program Run Time: '+extime)
    for i in range(len(POPULATION_SIZE)):
        fig,axs = plt.subplots(len(topchromo)+1,figsize=(20,20))
        fig.suptitle('Final Chromosomes With Population: '+str(POPULATION_SIZE[i])+timstr)
        disTARGET = chrom2img(TARGET, image_array.shape)
        axs[0].imshow(disTARGET)
        axs[0].set_title('TARGET  Num Pixels: '+str(len(TARGET)))
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)

        for k in range(len(topchromo)):
            generation = FinGen[k][i]
            tchor=topchromo[k]
            tpix=toppix[k]
            tfit=topfit[k]
            tpixper=toppixper[k]
            rt = time.strftime("%H:%M:%S", time.gmtime(finruntime[k][i]))
            disChrom = chrom2img(tchor[i], image_array.shape)
            axs[k+1].imshow(disChrom)
            axs[k+1].get_xaxis().set_visible(False)
            axs[k + 1].get_yaxis().set_visible(False)
            axs[k+1].set_title('Model: '+str(k+1)+" Gen: " + str(generation) + " Fit: " + str(
                    tfit[i])+" Correct Pix: "+str(tpix[i]) +" Goal Percentage: "+str(tpixper[i])+"%"+" Runtime: "+str(rt))
        plt.show()

        for i in range(len(POPULATION_SIZE)):
            fig, axs = plt.subplots(len(topchromo) + 1,figsize=(10,10))
            fig.suptitle('Final Chromosomes With Population: ' + str(POPULATION_SIZE[i]))
            disTARGET = chrom2img(TARGET, image_array.shape)
            axs[0].imshow(disTARGET)
            axs[0].set_title('TARGET')
            axs[0].get_xaxis().set_visible(False)
            axs[0].get_yaxis().set_visible(False)

            for k in range(len(topchromo)):
                generation = FinGen[k][i]
                tchor = topchromo[k]
                tpix = toppix[k]
                tfit = topfit[k]
                tpixper = toppixper[k]
                rt = time.strftime("%H:%M:%S", time.gmtime(finruntime[k][i]))
                disChrom = chrom2img(tchor[i], image_array.shape)
                axs[k + 1].imshow(disChrom)
                axs[k + 1].get_xaxis().set_visible(False)
                axs[k + 1].get_yaxis().set_visible(False)
                axs[k + 1].set_title('Model: ' + str(k + 1))
            plt.show()

    eltime = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(eltime)))

if __name__ == '__main__':
    main()
#
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
