import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys
from matplotlib.offsetbox import OffsetImage

info={0:1,34:2,38:3}#,35:4,36:6,38:3,39:5,40:7}


attributes={"rank1":["acc","1"],"rank10":["acc","10"],"corr":["collapse","|corr|"],"trivial_features":["collapse","trivial_features"]}

shape=(2,2)

dataset="metal"
metric="rank1"

df=np.load("datas/metal/one.npz")
full,reduced=df["full"],df["reduced"]


name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]

plt.figure(figsize=(10,5))


x,y,s=prepare_dic(dataset,info,*attributes[metric])
print(x,y,s)


dates=y

# Choose some nice levels
levels = np.tile([1/2],#, 5, -3, 3, -1, 1],
                 int(np.ceil(len(dates))))[:len(dates)]
levels = np.array([1/2,1/4,1/6])
alpha=0.42
levels = np.array([alpha,2*alpha/3,1*alpha/3])


# Create figure and plot a stem plot with the date
#fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")
#ax.set(title="Matplotlib release dates")

plt.gca().vlines(dates, 0, levels, color="black",linewidth=1.5)  # The vertical stems.
plt.plot(dates, np.zeros_like(dates), "o",
        color="k", markerfacecolor="w")  # Baseline and markers on it.
#plt.plot(dates,levels,"o",color="k",markerfacecolor="w")
#plt.plot(dates,levels,"h",color="red",markerfacecolor="red")

plt.ylim(bottom=0)
plt.xlim(0.45,1.00)



# annotate lines
#for d, l, r in zip(dates, levels, names):
#    plt.annotate(r, xy=(d, l),
#                xytext=(-3, np.sign(l)*3), textcoords="offset points",
#                horizontalalignment="right",
#                verticalalignment="bottom" if l > 0 else "top")

# format x-axis with 4-month intervals
#plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
#plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# remove y-axis and spines
plt.gca().yaxis.set_visible(False)
#plt.yticks([])
plt.gca().spines[["left","top","right"]].set_visible(False)
#plt.gca().twinx()
#plt.ylim(0,0.5)

plt.gca().margins(y=0.1)

aspect="equal"
aspect="auto"

def draw_one(rigth,lv,siz,annot,flip=False,ont=False):
    add=[0.1,0.1+2*siz]
    add=[lv-2*siz,lv]
    if flip:
        extend=[rigth,rigth+siz]+add
    else:
        extend=[rigth-siz,rigth]+add
    plt.imshow(full,extent=extend,origin="lower",aspect=aspect)#aspect="auto")
    motio=0.004
    bbox=dict(boxstyle="square",fc="white",color="black")
    if ont:
        plt.annotate(annot+"px",xy=(rigth + 0.66*(0.01 if flip else -0.01),lv+0.036+motio),xytext=(0,0),textcoords="offset points",horizontalalignment="left" if flip else "right",verticalalignment="bottom",color="black",bbox=bbox)
    else:
        plt.annotate(annot+"px",xy=(rigth + 1.5*(0.01)+motio,lv-0.02-siz+0.01),xytext=(0,0),textcoords="offset points",horizontalalignment="left",verticalalignment="bottom",color="black",bbox=bbox)

    if ont:
        delta=0.006+motio
        xp=rigth-siz/2*(-1 if flip else 1)
        yp=lv-siz*2-delta if ont==False else lv+delta
        plt.annotate('', xy=(xp, yp), xytext=(xp, yp-(-0.02 if ont else 0.02)),# xycoords='axes fraction', 
                    ha='left', va='top',
                    bbox=None,#dict(boxstyle='square', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=f'-[, widthB={5*4.60*siz}, lengthB=0.2', lw=1.0, color='black',alpha=1.0))
    else:
        delta=0.002+motio
        xp=rigth+delta#-siz/2*(-1 if flip else 1)
        yp=lv-siz if ont==False else lv+delta
        plt.annotate('', xy=(xp, yp), xytext=(xp+0.009, yp),# xycoords='axes fraction', 
                    ha='left', va='top',
                    bbox=None,#dict(boxstyle='square', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=f'-[, widthB={4.60*lv}, lengthB=0.2', lw=1.0, color='black',alpha=1.0))
    

#draw_one(y[0],0.1,r"$2^{16}$")
#draw_one(y[1],0.1/2,r"$\frac{2^{16}}{4}$")
#draw_one(y[2],0.1/3,r"$\frac{2^{16}}{9}$",flip=True)

#draw_one(y[0],0.1,r"$65536$")
#draw_one(y[1],0.1/2,r"$16384$")
#draw_one(y[2],0.1/3,r"$7225$",flip=True)

draw_one(y[0],levels[0],0.1,r"$256$",ont=True)
draw_one(y[1],levels[1],0.1/2,r"$128$",ont=True)
draw_one(y[2],levels[2],0.1/3,r"$85$",flip=True,ont=True)


#xlim=plt.gca().get_xlim()
#ylim=plt.gca().get_ylim()
#plt.imshow(full,cmap="gray",vmin=0,vmax=1,extent=[0.7,0.3,0.2,0.4],aspect=3/1)
#plt.gca().set_xlim(xlim)
#plt.gca().set_ylim(ylim)

plt.xlabel("Rank-1 Accuracy")
xt=[0.5,0.6,0.7,0.8,0.9,1.0]
xl=[r"$%d\%%$"%int(x*100) for x in xt]
plt.xticks(xt,xl)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off


col1="black"
mov=0.045
#draw arrow from left to right
plt.annotate("",xy=(0.62,0.50+mov),xytext=(0.91,0.50+mov),arrowprops=dict(arrowstyle="<-",color=col1))
#label arrow
plt.annotate("Better Re-identification",xy=(0.630,0.515+mov),xytext=(0,0),textcoords="offset points",horizontalalignment="left",verticalalignment="bottom",color=col1)

col2="black"
mov=0.06
m2=-0.03
#draw arrow from top to bottom
plt.annotate("",xy=(0.94+mov,0.54+m2),xytext=(0.94+mov,0.07+m2),arrowprops=dict(arrowstyle="<-",color=col2))
#label arrow
plt.annotate("Fewer Features",xy=(0.935+mov,0.125+m2),xytext=(0,0),textcoords="offset points",horizontalalignment="right",verticalalignment="bottom",color=col2,rotation=90)

#plt.ylim(top=0.55)
plt.ylim(top=0.62)


plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")



plt.show()




