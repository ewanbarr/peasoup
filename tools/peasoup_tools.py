import os
from popen2 import popen2
from sigpyproc.Readers import FilReader
import glob
import sys
import gzip
import numpy as np
import pylab as plt
from lxml import etree
from matplotlib import mlab
from struct import pack,unpack
import time

def radec_to_str(val):
    if val < 0:
        sign = -1
    else:
        sign = 1
    fractional,integral = np.modf(abs(val))
    xx = (integral-(integral%10000))/10000
    yy = ((integral-(integral%100))/100)-xx*100
    zz = integral - 100*yy - 10000*xx + fractional
    zz = "%07.4f"%(zz)
    return "%02d:%02d:%s"%(sign*xx,yy,zz)


class Candidate(object):
    def __init__(self,cand_dict,fold,hits):
        for key,value in cand_dict.items():
            setattr(self,key,value)
        self.fold = fold
        self.hits = hits
        

class PeasoupOutput(object):
    def __init__(self, overview_file, candidate_file):
        self._xml_parser = OverviewFile(overview_file)
        self._cand_parser = CandidateFileParser(candidate_file)

    def get_candidate(self, idx):
        cand_dict = self._xml_parser.get_candidate(idx)
        fold,hits = self._cand_parser.cand_from_offset(cand_dict["byte_offset"])
        return Candidate(cand_dict,fold,hits)


class CandidateFileParser(object):
    _dtype = [("dm","float32"),
              ("dm_idx","int32"),
              ("acc","float32"),
              ("nh","int32"),
              ("snr","float32"),
              ("freq","float32")]

    def __init__(self, filename):
        self._f = open(filename,"r")
        
    def _read_fold(self):
        nbins,nints = unpack("II",self._f.read(8))
        fold = np.fromfile(self._f,dtype="float32",count=nbins*nints)
        fold = fold.reshape(nints,nbins)
        return fold
        
    def _read_hits(self):
        count, = unpack("I",self._f.read(4))
        cands = np.fromfile(self._f,dtype=self._dtype,count=count)
        return cands

    def cand_from_offset(self, offset):
        self._f.seek(offset)
        if self._f.read(4) == "FOLD":
            fold = self._read_fold()
            hits = self._read_hits()
            return fold,hits
        else:
            self._f.seek(offset)
            hits = self._read_hits()
            return None,hits
    
    def __del__(self):
        self._f.close()
    

class OverviewFile(object):
    _ar_dtype = [('period','float32'),
                 ('dm','float32'),
                 ('acc','float32'),
                 ('nh','float32'),
                 ('snr','float32')]
                 
    _dtype = [
        ('cand_num','int32'),
        ('period','float32'),
        ('opt_period','float32'),
        ('dm','float32'),
        ('acc','float32'),
        ('nh','float32'),
        ('snr','float32'),
        ('folded_snr','float32'),
        ('is_adjacent','ubyte'),
        ('is_physical','ubyte'),
        ('ddm_count_ratio','float32'),
        ('ddm_snr_ratio','float32'),
        ('nassoc','int32'),
        ('byte_offset',"int32")]
    
    def __init__(self,name):
        with open(name,"r") as f:
            xml_string = f.read()
            
        try:
            self._xml = etree.fromstring(xml_string)
        except etree.XMLSyntaxError:
            start = "<username>"
            end = "</username>"
            start_idx = xml_string.find(start)+len(start)
            end_idx = xml_string.find(end)
            new_xml_string = "pulsar".join([ xml_string[:start_idx] , xml_string[end_idx:] ])
            self._xml = etree.fromstring(new_xml_string)

        self._candidates = self._xml.find("candidates").findall("candidate")
        self._ncands = len(self._candidates)
            
    def __str__(self):
        return etree.tostring(self._xml,pretty_print=True)
        
    def as_array(self):
        cands = np.recarray(self._ncands,dtype=self._dtype)
        for cand,candidate in zip(cands,self._candidates):
            cand["cand_num"] = candidate.attrib["id"]
            for tag,typename in self._dtype:
                if tag == "cand_num":
                    cand["cand_num"] = candidate.attrib["id"]
                else:
                    cand[tag] = candidate.find(tag).text
                              
        return cands

    def get_candidate(self,idx):
        cand_dict = {}
        cand = self._candidates[idx]
        for tag,typename in self._dtype:
            if tag == "cand_num":
                value = cand.attrib["id"]
            else:
                value = cand.find(tag).text
            cand_dict[tag] = np.asscalar(np.array([value]).astype(typename))
        return cand_dict
    
    def get_candidate_data(self,idx):
        cand_dict = self.get_candidate(idx)
        return CandidateFileParser("candidates.peasoup")

    def make_predictor(self,idx):
        cand = self.get_candidate(idx)
        header = self._xml.find("header_parameters")
        ra = radec_to_str(float(header.find("src_raj").text))
        dec = radec_to_str(float(header.find("src_dej").text))
        predictor = ("SOURCE: %s"%header.find("source_name").text,
                     "PERIOD: %.15f"%cand["period"],
                     "DM: %.3f"%cand["dm"],
                     "ACC: %.3f"%cand["acc"],
                     "RA: %s"%ra,
                     "DEC: %s"%dec)
        return "\n".join(predictor)

      
class CandidatePlotter(object):
    def __init__(self,candidate):
        self.candidate = candidate
        self.fig = plt.figure(figsize=[14,12])
        self.prof_ax = plt.subplot2grid([5,9],[0,1],colspan=2)
        self.fold_ax = plt.subplot2grid([5,9],[1,1],colspan=2,rowspan=2,sharex=self.prof_ax)
        self.subs_ax = plt.subplot2grid([5,9],[1,0],rowspan=2,sharey=self.fold_ax)
        self.table_ax = plt.subplot2grid([5,9],[0,3],colspan=3,rowspan=3,frameon=False)
        self.dm_ax  = plt.subplot2grid([5,9],[0,6],colspan=2)
        self.acc_ax = plt.subplot2grid([5,9],[1,8],colspan=1,rowspan=2)
        self.dm_acc_ax = plt.subplot2grid([5,9],[1,6],colspan=2,rowspan=2,sharex=self.dm_ax,sharey=self.acc_ax)
        self.all_ax = plt.subplot2grid([6,9],[4,0],colspan=9,rowspan=3)
        self._plot_all_cands(self.all_ax)
        self.timers = {
            "read":Timer(),
            "prof":Timer(),
            "fold":Timer(),
            "stat":Timer(),
            "table":Timer(),
            "dm":Timer(),
            "acc":Timer(),
            "dmacc":Timer(),
            "write":Timer(),
            "clear":Timer()
            }
        self.header = self.candidate
        #self.fig.suptitle("Source name: %s"%self.header.source_name,fontsize=16)

    def _plot_all_cands(self,ax):
        ar = self.candidate.hits
        ax.set_xscale("log")
        """
        high_snr = ar[np.where(ar["snr"]>40.0)]
        low_snr = ar[np.where(ar["snr"]<=40.0)]
        low_snr_sizes = low_snr["snr"]
        low_snr_sizes-= low_snr_sizes.min()
        low_snr_sizes/= low_snr_sizes.max()
        low_snr_sizes*= 250
        low_snr_sizes+= 5
        ax.scatter(low_snr["period"],low_snr["dm"],c=low_snr["nh"],s=low_snr_sizes)
        ax.scatter(high_snr["period"],high_snr["dm"],c=high_snr["nh"],s=high_snr["snr"],marker="x")
        """
        ax.scatter(1/ar["freq"],ar["dm"],ar["snr"])
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("DM (pccm^-3)")
        ax.set_xlim(1/(ar["freq"]).min(),(1/ar["freq"]).max())
        ax.set_ylim(ar["dm"].min(),ar["dm"].max())
        self.xline = ax.vlines(0,0,0)
        self.yline = ax.hlines(0,0,0)
        self.xline_lims = ax.get_ylim()
        self.yline_lims = ax.get_xlim()
        
    def _set_crosshair(self,x,y):
        self.xline.remove()
        self.yline.remove()
        self.xline = self.all_ax.vlines(x,self.xline_lims[0],self.xline_lims[1])
        self.yline = self.all_ax.hlines(y,self.yline_lims[0],self.yline_lims[1])
        
    def clear_all(self):
        self.prof_ax.cla()
        self.fold_ax.cla()
        self.subs_ax.cla()
        self.table_ax.cla()
        self.dm_ax.cla()
        self.acc_ax.cla()
        self.dm_acc_ax.cla()

    def _fill_table(self,ax,header):
        self._set_crosshair(self.candidate.period,self.candidate.dm)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ra = radec_to_str(float(header.find("src_raj").text))
        dec = radec_to_str(float(header.find("src_dej").text))
        info = (("R.A.",ra), 
                ("Decl.",dec), 
                ("P0",stats["period"]), 
                ("Opt P0",stats["opt_period"]),
                ("DM","%.2f"%stats["dm"]),
                ("Acc",stats["acc"]),
                ("Harmonic",stats["nh"]),
                ("Spec S/N","%.1f"%stats["snr"]),
                ("Fold S/N","%.1f"%stats["folded_snr"]),
                ("Adjacent?",bool(stats["is_adjacent"])),
                ("Physical?",bool(stats["is_physical"])),
                ("DDM ratio 1",stats["ddm_count_ratio"]),
                ("DDM ratio 2",stats["ddm_snr_ratio"]),
                ("Nassoc",stats["nassoc"]))
        cell_text = [(val[0],val[1]) for val in info]
        tab = ax.table(cellText=cell_text,cellLoc="left",
                                  colLoc='left',loc="center")
        tprops = tab.properties()
        leftcol = [val[0] for val in info]
        for cell in tprops["child_artists"]:
            if cell.get_text().get_text() in leftcol:
                cell.set_width(0.4)
            else:
                cell.set_width(0.5)
            cell.set_linewidth(0)

        tab.scale(1.0,2.0)
        tab.set_fontsize(30)

    def _plot_subints(self,ax,ar):
        ax.imshow(ar,aspect="auto",interpolation="nearest")
        ax.set_xlim(-0.5,ar.shape[1]-0.5)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel("Phase bin")

    def _plot_profile(self,ax,ar):
        ax.plot(ar.sum(axis=0))
        ax.set_ylabel("Flux")
        ax.set_title("Profile")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    def _plot_subint_stats(self,ax,ar):
        ydata = range(ar.shape[0])
        fstd = ar.std(axis=1)
        y0 = ar.mean(axis=1)-3*fstd
        y1 = ar.mean(axis=1)+3*fstd
        ax.fill_betweenx(ydata,y0,y1,alpha=0.5,color="lightblue",label="+-3 sigma")
        ax.plot(ar.mean(axis=1),ydata,lw=2,alpha=0.8,color="lightblue",label="mean")
        ax.plot(ar.min(axis=1),ydata,lw=2,c="darkblue",label="min")
        ax.plot(ar.max(axis=1),ydata,lw=2,c="darkred",label="max")
        ax.legend(loc='lower left', bbox_to_anchor=(-0.2, 1.0),prop={'size':10})
        m1,m2 = ax.get_xlim()
        ax.set_xlim(m2,m1)
        ax.set_ylim(-0.5,ar.shape[0]-0.5)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylabel("Subintegration")

    def _plot_acc_dm_map(self,ax,cand,limits):
        #xi = np.linspace(limits[0],limits[1],2*np.unique(cand["dm"]).size)
        #yi = np.linspace(limits[3],limits[2],2*np.unique(cand["acc"]).size)
        #dm_acc = mlab.griddata(cand["dm"],cand["acc"],cand["snr"],xi,yi)
        
        harms = np.unique(cand["nh"])
        colors = ["darkblue","lightblue","green","orange","darkred"]
        snrs = np.copy(cand["snr"])
        for ii,harm in enumerate(harms):
            idxs = np.where(cand["nh"]==harm)
            subcand = cand[idxs]
            sizes = snrs[idxs]
            sizes-= sizes.min()
            sizes/= sizes.max()
            sizes*= 250
            sizes+= 5
            ax.scatter(subcand["dm"],subcand["acc"],facecolor=colors[ii],edgecolor="none",s=sizes)
            
        #ax.imshow(dm_acc,aspect="auto",extent=limits,interpolation="nearest")
        #ax.contour(dm_acc[::-1],extent=limits)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlim(limits[0],limits[1])
        ax.set_ylim(limits[2],limits[3])
        ax.set_xlabel("DM (pc cm^-3)")
        
    def _plot_acc_scatter(self,ax,cand):
        ax.yaxis.tick_right()
        harms = np.unique(cand["nh"])
        colors = ["darkblue","lightblue","green","orange","darkred"]
        for ii,harm in enumerate(harms):
            subcand = cand[np.where(cand["nh"]==harm)]
            ax.scatter(subcand["snr"],subcand["acc"],edgecolor="none",facecolor=colors[ii],label="Harm. %d"%harm)
        ax.legend(loc='lower left', bbox_to_anchor=(0.2, 1.0),prop={'size':10})
        for label in ax.get_xticklabels():
            label.set_rotation(-45)
        ax.set_ylabel("Accleration (m/s/s)",rotation=-90)
        ax.yaxis.set_label_position("right")
        ax.set_xlabel("S/N")

    def _plot_dm_scatter(self,ax,cand):
        ax.yaxis.tick_right()
        harms = np.unique(cand["nh"])
        colors = ["darkblue","lightblue","green","orange","darkred"]
        for ii,harm in enumerate(harms):
            subcand = cand[np.where(cand["nh"]==harm)]
            ax.scatter(subcand["dm"],subcand["snr"],facecolor=colors[ii],edgecolor="none",label="Harm. %d"%harm)
        ax.set_ylabel("S/N",rotation=-90)
        ax.yaxis.set_label_position("right")
        plt.setp(ax.get_xticklabels(), visible=False)

    def plot_cand(self,filename=None):

        self.timers["read"].start()
        fold = self.candidate.fold
        if fold is not None:
            fold -= fold.min()
            fold /= fold.max()
        cand = np.sort(self.candidate.hits,order="snr")[::-1]
        limits = [cand["dm"].min(),cand["dm"].max(),cand["acc"].min(),cand["acc"].max()]
        self.timers["read"].stop()

        self.timers["clear"].start()
        self.clear_all()
        self.timers["clear"].stop()
        
        self.timers["prof"].start()
        if fold is not None:
            self._plot_profile(self.prof_ax,fold)
            self._plot_subints(self.fold_ax,fold)
            self._plot_subint_stats(self.subs_ax,fold)
        
        #self._fill_table(self.table_ax,self.header)
        if self.candidate.period < 0.1:
            self._plot_acc_dm_map(self.dm_acc_ax,cand,limits)
            self._plot_acc_scatter(self.acc_ax,cand)
        
        self._plot_dm_scatter(self.dm_ax,cand)
        self.timers["prof"].stop()

        self.timers["write"].start()
        if filename is None:
            plt.draw()
        else:
            self.fig.savefig(filename)
        
        self.timers["write"].stop()

class Timer(object):
    def __init__(self):
        self.elapsed = 0.0
        self.started = None

    def start(self):
        self.started = time.time()

    def stop(self):
        if self.started is not None:
            self.elapsed += time.time() - self.started
        self.started = None

def main(filename):
    x = OverviewFile(filename)
    z = CandidatePlotter(x)
    for ii in range(100):
        try:
            z.plot_cand(ii,"Cand%04d.png"%ii)
        except Exception as error:
            print error

    for name,timer in z.timers.items():
        print name,timer.elapsed/ii


if __name__ == "__main__":
    main(sys.argv[1])
    


