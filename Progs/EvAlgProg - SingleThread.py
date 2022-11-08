# #Imports
import functools
import operator
import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from time import sleep
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2


def img2chromo(img_arr):
    chromosome = np.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))
    return chromosome

def chrom2img(chromosome, img_shape):
    img_arr = np.reshape(a=chromosome, newshape=img_shape)
    return img_arr


# User Adjustments
POPULATION_SIZE = [50,200]##number of Chromosomes ,500,2000
MaxGen = 1500
Pc = .9995 #porbability of crossover, the left over .1 is chance at mutation
El = .1 #Percentage Elite
MutAd = .005 #when our fitness has less then a .5% change from the previous best fitness, increase our mutation
FM = 1 #Which fitness method to use, 1:mean, 2: number difference, 3:non correct pixels
imgCreate = 1500 #create new image every X generations
image = Image.open("IMG1.jpg") #select Image to be the target

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

def mse(IMG1,IMG2):
    err = np.sum((IMG1.astype("float") - IMG2.astype("float")) ** 2)
    err /= float(IMG1.shape[0] * IMG1.shape[1])
    return err
# Target string to be generated
TARGET = IMG_Arr2

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
        gnome_len = GENES(TARGET.shape)
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
            count = np.mean(np.abs(TARGET-self.chromosome))
            fitness+=count
        elif FM==2:
            #Method2 - get the absolute difference
            #res = np.abs(self.chromosome - TARGET)
            res = np.abs(TARGET - self.chromosome)
            count = np.sum(res)
            fitness+=count
        elif FM==3:
            #Method3 - only increase fitness if pixel not correct
            count = np.count_nonzero(np.abs(TARGET-self.chromosome))
            fitness += count
            fitness = np.int64(fitness)

        return fitness

    def cal_pixscore(self):
        incorrect = np.count_nonzero(np.abs(TARGET-self.chromosome))
        pixscore = len(TARGET)-incorrect
        return pixscore

    def pixclosesc(self):
        res2 = np.abs(TARGET - self.chromosome)
        difcount = int(np.sum(res2))
        tarcount = int(np.sum(TARGET))
        countres = tarcount-difcount
        if countres<=0:countres=0
        res = (countres/tarcount)*100
        res = round(res, 5)
        return res

#Create new child
def CreateChild(s,cumsum,population):
    new_generation=[]
    for _ in range(s):
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
    return new_generation




# # Driver code
def main():
    global POPULATION_SIZE
    st = time.time()

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
    print('Creating Population')
    for popsize in POPULATION_SIZE:
        printpoint = popsize/10
        pipcount=0
        population = []
        for _ in range(popsize):
            gnome = Individual.create_gnome()
            population.append(Individual(gnome))
            if pipcount==printpoint:
                print(str(_) +'/'+str(popsize))
                pipcount=0
            pipcount+=1
        initalpop.append(population)
    totfit=[]
    totpix=[]
    topchromo=[]
    FinGen = []  # final Generation
    totimags=[]
    finruntime=[]

    for fm in range(1,4):
        substart=time.time()
        global FM
        FM=fm
        corpixnum=[]
        fitscrOT=[]
        bchromo = []  # save best chromosome
        fingen = []
        imagsmOT=[]
        for population in initalpop:
            newpop = []
            for peep in population:
                newpop.append(Individual(peep.chromosome))
            population=newpop
            MGRun = tqdm(range(MaxGen))
            fitscrOTTMP=[]
            corpixnumTMP=[]
            imagsmOTTMP=[]
            # current generation
            generation = 0
            for _ in MGRun:
                # sort the population in increasing order of fitness score
                population = sorted(population, key=lambda x: x.fitness)
                sleep(.01)
                MGRun.set_description("STP: "+str(fm) +" Pop: %s" % str(len(population)) +" Fitness: %s" % str(population[0].fitness)+" PixelID: %s" % str(population[0].pixclosesc))

                # if the individual having lowest fitness score ie.
                # 0 then we know that we have reached to the target
                # and break the loop
                fitscrOTTMP.append(population[0].fitness)
                corpixnumTMP.append(population[0].pixscore)
                imagsmOTTMP.append(population[0].pixclosesc)

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
                #threading
                subs = int(s/10)

                for _ in range(s):
                    R = np.random.rand()
                    if R<cumsum[-1]:R=cumsum[-1]
                    p1Index, = np.where(cumsum<=R)
                    parent1 = population[p1Index[0]]
                    #parent1 = np.random.choice(population[:halfpop],p=cumsum[:halfpop])
                    R = np.random.rand()
                    if R<cumsum[-1]:R=cumsum[-1]
                    p2Index, = np.where(cumsum<=R)
                    parent2 = population[p2Index[0]]
                    #parent2 = np.random.choice(population[:halfpop],p=cumsum[:halfpop])
                    child = parent1.mate(parent2)
                    new_generation.append(child)

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
                    ax[0].set_title('TARGET Num Pixels: '+str(len(TARGET)))
                    #ax[1].imshow(population[0].chromosome,interpolation='nearest')
                    disChrom = chrom2img(population[0].chromosome, image_array.shape)
                    ax[1].imshow(disChrom)
                    ax[1].set_title('Best Chromosome')
                    plt.suptitle("MTH: "+str(fm)+" Pop:"+str(len(population))+" Gen: "+str(generation)+" Fit: "+str(population[0].fitness)+" CP: "+str(population[0].pixscore)+" GP: "+str(population[0].pixclosesc)+"%")
                    plt.show()
                    bestfit = population[0].fitness
                generation += 1

            corpixnum.append(corpixnumTMP)
            fitscrOT.append(fitscrOTTMP)
            bchromo.append(population[0])
            fingen.append(generation)
            imagsmOT.append(imagsmOTTMP)
        topchromo.append(bchromo)
        totfit.append(fitscrOT)
        totpix.append(corpixnum)
        FinGen.append(fingen)
        totimags.append(imagsmOT)
        subel = time.time()-substart
        finruntime.append(subel)

    #Graph of fitness over generations
    figsiz = (15,15)
    fig, axs1 = plt.subplots(len(totfit), figsize=figsiz) #FITNESS V GEN
    for j in range(len(totfit)):
        fit = totfit[j]
        #plt.figure()
        for i in range(len(fit)):
            axs1[j].plot(range(100,len(fit[i])),fit[i][100:],label='Pop: '+str(POPULATION_SIZE[i]))
        # plt.xlabel("Generations")
        # plt.ylabel("Fitness Score")
        axs1[j].set_title("Fitness over Generation for Fitness Model "+str(j+1))
        axs1[j].legend(loc="upper right")
        # plt.legend()
        # plt.show()
    for ax in axs1.flat:
        ax.set(xlabel='Generations', ylabel='Fitness Score')
    for ax in axs1.flat:
        ax.label_outer()

    fig, axs2 = plt.subplots(len(totpix), figsize=figsiz) #CORRECT PIX
    for j in range(len(totpix)):
        pix = totpix[j]
        #plt.figure()
        for i in range(len(pix)):
            axs2[j].plot(range(len(pix[i])),pix[i],label='Pop: '+str(POPULATION_SIZE[i]))
        # plt.xlabel("Generations")
        # plt.ylabel("Correct Pixel Count")
        axs2[j].set_title("Correct Pixels over Generation for Fitness Model "+str(j+1))
        axs2[j].legend(loc="lower right")
        # plt.legend()
        # plt.show()
    for ax in axs2.flat:
        ax.set(xlabel='Generations', ylabel='Correct Pixel Count')
    for ax in axs2.flat:
        ax.label_outer()

    # fig, axs3 = plt.subplots(len(totfit), figsize=figsiz)
    # plt.figure()
    # for i in range(len(totfit)):
    #     tfit=totfit[i]
    #     res = tfit[-1]
    #     plt.plot(range(100,len(res)),res[100:],label='FitMod:'+str(i+1)+' Pop: '+str(POPULATION_SIZE[-1]))
    # plt.xlabel("Generations")
    # plt.ylabel("Fitness")
    # plt.title("Fitness over Generation")
    # plt.legend()
    # plt.show()

    plt.figure()
    for i in range(len(totpix)):
        tfit=totpix[i]
        res = tfit[-1]
        plt.plot(range(len(res)),res,label='FitMod:'+str(i+1)+' Pop: '+str(POPULATION_SIZE[-1]))
    plt.xlabel("Generations")
    plt.ylabel("Correct Pixel Count")
    plt.title("Correct Pixel Count over Generation")
    plt.legend()
    plt.show()

    plt.figure()
    for i in range(len(totimags)):
        tfit=totimags[i]
        res = tfit[-1]
        plt.plot(range(len(res)),res,label='FitMod:'+str(i+1)+' Pop: '+str(POPULATION_SIZE[-1]))
    plt.xlabel("Generations")
    plt.ylabel("Percentage Identical to Original")
    plt.title("Percent Identical to Original over Generation")
    plt.legend()
    plt.show()

    eltime = time.time() - st
    extime = time.strftime("%H:%M:%S", time.gmtime(eltime))
    timstr = str(' Total Program Run Time: '+extime)
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
            rt = time.strftime("%H:%M:%S", time.gmtime(finruntime[k]))
            tchor=topchromo[k]
            disChrom = chrom2img(tchor[i].chromosome, image_array.shape)
            axs[k+1].imshow(disChrom)
            axs[k+1].get_xaxis().set_visible(False)
            axs[k + 1].get_yaxis().set_visible(False)
            axs[k+1].set_title('Model: '+str(k+1)+" Gen: " + str(generation) + " Fit: " + str(
                    tchor[i].fitness)+" Correct Pix: "+str(tchor[i].pixscore)+" Goal Percentage: "+str(tchor[i].pixclosesc)+"%"+" Runtime: "+str(rt))
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
            rt = time.strftime("%H:%M:%S", time.gmtime(finruntime[k]))
            tchor = topchromo[k]
            disChrom = chrom2img(tchor[i].chromosome, image_array.shape)
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
