# This script is written based on DeepFRI evaluation.py https://github.com/flatironinstitute/DeepFRI
import csv
import pickle as pkl
import obonet
import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score as aupr

go_graph = obonet.read_obo(open("/root/autodl-tmp/data/go-basic.obo", "r"))


def bootstrap(Y_true, Y_pred):
    n = Y_true.shape[0]
    idx = np.random.choice(n, n)
    return Y_true[idx], Y_pred[idx]


def load_test_prots(fn):
    proteins = []
    seqid_mtrx = []
    with open(fn, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            inds = row[1:]
            inds = np.asarray([int(i) for i in inds]).reshape(1, len(inds))
            proteins.append(row[0])
            seqid_mtrx.append(inds)
    return np.asarray(proteins), np.concatenate(seqid_mtrx, axis=0)


def load_go2ic_mapping(fn):
    goterm2ic = {}
    with open(fn, 'r') as f_read:
        for line in f_read:
            goterm, ic = line.strip().split()
            goterm2ic[goterm] = float(ic)
    return goterm2ic


def propagate_go_preds(Y_hat, goterms):
    """Propagate GO prediction scores upward to ancestors (max)."""
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm in go_graph:
            parents = set(goterms).intersection(nx.descendants(go_graph, goterm))
            for parent in parents:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]], Y_hat[:, go2id[parent]])
    return Y_hat


def propagate_ec_preds(Y_hat, goterms):
    """Propagate EC scores from specific a.b.c.d to wildcard parent a.b.c.- (max)."""
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm.find('-') == -1:
            parent = goterm.split('.')
            parent[-1] = '-'
            parent = ".".join(parent)
            if parent in go2id:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]], Y_hat[:, go2id[parent]])
    return Y_hat


def build_descendants_cache(goterms):
    """
    Precompute all descendants (full GO graph, not truncated to label set) for each goterm.
    Returns a list aligned with goterms order.
    """
    cache = []
    for go in goterms:
        if go in go_graph:
            cache.append(list(nx.descendants(go_graph, go)))
        else:
            cache.append([])
    return cache


def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False, returnRuMi=False):
    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru ** 2 + mi ** 2)
    if avg:
        ru = np.mean(ru)
        mi = np.mean(mi)
        sd = np.sqrt(ru ** 2 + mi ** 2)
    if not returnRuMi:
        return sd
    return [ru, mi, sd]


def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nru = num / denom
    if avg:
        nru = np.mean(nru)
    return nru


def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nmi = num / denom
    if avg:
        nmi = np.mean(nmi)
    return nmi


# Load precomputed IC from training set counts
with open("/root/autodl-tmp/data/ic_count.pkl", "rb") as f:
    ic_count = pkl.load(f)
    ic_count['bp'] = np.where(ic_count['bp'] == 0, 1, ic_count['bp'])
    ic_count['mf'] = np.where(ic_count['mf'] == 0, 1, ic_count['mf'])
    ic_count['cc'] = np.where(ic_count['cc'] == 0, 1, ic_count['cc'])
    train_ic = {
        'bp': -np.log2(ic_count['bp'] / 69709),
        'mf': -np.log2(ic_count['mf'] / 69709),
        'cc': -np.log2(ic_count['cc'] / 69709),
    }


class Method(object):
    def __init__(self, method_name, annot_dict):
        """
        annot_dict keys: Y_true, Y_pred, goterms, gonames (optional), proteins, ontology ('bp'/'mf'/'cc' or 'ec')
        """
        self.method_name = method_name
        self.Y_true = annot_dict['Y_true']
        self.Y_pred = annot_dict['Y_pred']
        self.goterms = annot_dict['goterms']
        self.gonames = annot_dict.get('gonames', None)
        self.proteins = annot_dict['proteins']
        self.ont = annot_dict['ontology']
        # cache descendants before evaluation
        self._descendants = build_descendants_cache(self.goterms)
        self._propagate_preds()
        if self.ont == 'ec':
            goidx = [i for i, goterm in enumerate(self.goterms) if goterm.find('-') == -1]
            self.Y_true = self.Y_true[:, goidx]
            self.Y_pred = self.Y_pred[:, goidx]
            self.goterms = [self.goterms[idx] for idx in goidx]
            if self.gonames:
                self.gonames = [self.gonames[idx] for idx in goidx]
            # filter descendants to remaining labels
            keep = set(self.goterms)
            self._descendants = [[d for d in desc if d in keep] for desc in self._descendants if desc is not None]
        self.termIC = train_ic[self.ont]
        self.ic_bins_3= None
        
    def get_ic_bins(self):
        ic = np.asarray(self.termIC)
        valid = np.isfinite(ic)
        
        self.ic_bins_3 = {
            "IC<5": np.where((ic < 5) & valid)[0],
            "5<=IC<10": np.where((ic >= 5) & (ic < 10) & valid)[0],
            "IC>=10": np.where((ic >= 10) & valid)[0],
        }
        return self.ic_bins_3
        

    def _propagate_preds(self):
        if self.ont == 'ec':
            self.Y_pred = propagate_ec_preds(self.Y_pred, self.goterms)
        else:
            self.Y_pred = propagate_go_preds(self.Y_pred, self.goterms)

    def _cafa_ec_aupr(self, labels, preds):
        n = labels.shape[0]
        goterms = np.asarray(self.goterms)
        prot2goterms = {}
        for i in range(0, n):
            prot2goterms[i] = set(goterms[np.where(labels[i] == 1)[0]])
        F_list, AvgPr_list, AvgRc_list, thresh_list = [], [], [], []
        for t in range(1, 100):
            threshold = t / 100.0
            predictions = (preds > threshold).astype(np.int64)
            m = 0
            precision = 0.0
            recall = 0.0
            for i in range(0, n):
                pred_gos = set(goterms[np.where(predictions[i] == 1)[0]])
                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0:
                    m += 1
                    precision += float(num_overlap) / num_pred
                if num_true > 0:
                    recall += float(num_overlap) / num_true
            if m > 0:
                AvgPr = precision / m
                AvgRc = recall / n
                if AvgPr + AvgRc > 0:
                    F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)
        return np.asarray(AvgRc_list), np.asarray(AvgPr_list), np.asarray(F_list), np.asarray(thresh_list)

    def _cafa_go_aupr(self, labels, preds):
        # expand true/pred labels to all descendants (cached), remove ontology root
        n = labels.shape[0]
        ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}
        root = ont2root[self.ont]
        prot2goterms = {}
        for i in range(0, n):
            all_gos = set()
            for goidx in np.where(labels[i] == 1)[0]:
                all_gos.add(self.goterms[goidx])
                for child in self._descendants[goidx]:
                    all_gos.add(child)
            all_gos.discard(root)
            prot2goterms[i] = all_gos
        F_list, AvgPr_list, AvgRc_list, thresh_list = [], [], [], []
        for t in range(1, 100):
            threshold = t / 100.0
            predictions = (preds > threshold).astype(np.int64)
            m = 0
            precision = 0.0
            recall = 0.0
            for i in range(0, n):
                pred_gos = set()
                for goidx in np.where(predictions[i] == 1)[0]:
                    pred_gos.add(self.goterms[goidx])
                    for child in self._descendants[goidx]:
                        pred_gos.add(child)
                pred_gos.discard(root)
                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0 and num_true > 0:
                    m += 1
                    precision += float(num_overlap) / num_pred
                    recall += float(num_overlap) / num_true
            if m > 0:
                AvgPr = precision / m
                AvgRc = recall / n
                if AvgPr + AvgRc > 0:
                    F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)
        return np.asarray(AvgRc_list), np.asarray(AvgPr_list), np.asarray(F_list), np.asarray(thresh_list)

    def _function_centric_aupr(self, keep_pidx=None, keep_goidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        if keep_goidx is not None:
            tmp = []
            for goidx in keep_goidx:
                if Y_true[:, goidx].sum() > 0:
                    tmp.append(goidx)
            keep_goidx = tmp
        else:
            keep_goidx = np.where(Y_true.sum(axis=0) > 0)[0]
        Y_true = Y_true[:, keep_goidx]
        Y_pred = Y_pred[:, keep_goidx]
        micro_aupr = aupr(Y_true, Y_pred, average='micro')
        macro_aupr = aupr(Y_true, Y_pred, average='macro')
        aupr_goterms = aupr(Y_true, Y_pred, average=None)
        return micro_aupr, macro_aupr, aupr_goterms

    def _protein_centric_fmax(self, keep_pidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        if self.ont in {'mf', 'bp', 'cc'}:
            Recall, Precision, Fscore, thresholds = self._cafa_go_aupr(Y_true, Y_pred)
        else:
            Recall, Precision, Fscore, thresholds = self._cafa_ec_aupr(Y_true, Y_pred)
        return Fscore, Recall, Precision, thresholds

    def fmax(self, keep_pidx=None):
        fscore, _, _, _ = self._protein_centric_fmax(keep_pidx=keep_pidx)
        return max(fscore) if len(fscore) > 0 else 0.0

    def macro_aupr(self, keep_pidx=None, keep_goidx=None):
        _, macro_aupr, _ = self._function_centric_aupr(keep_pidx=keep_pidx, keep_goidx=keep_goidx)
        return macro_aupr
    
    def micro_aupr(self, keep_pidx=None, keep_goidx=None):
        micro_aupr, _, _ = self._function_centric_aupr(keep_pidx=keep_pidx, keep_goidx=keep_goidx)
        return micro_aupr

    def protein_centric_aupr(self, keep_pidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        if self.ont in {'mf', 'bp', 'cc'}:
            Recall, Precision, _, _ = self._cafa_go_aupr(Y_true, Y_pred)
        else:
            Recall, Precision, _, _ = self._cafa_ec_aupr(Y_true, Y_pred)
        if len(Recall) == 0 or len(Precision) == 0:
            return 0.0
        idx = np.argsort(Recall)
        return float(np.trapz(Precision[idx], Recall[idx]))

    def smin(self, keep_pidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        nrThresholds = 100
        thresholds = np.linspace(0.0, 1.0, nrThresholds)
        ss = np.zeros(thresholds.shape)
        for i, t in enumerate(thresholds):
            ss[i] = normalizedSemanticDistance(Y_true, (Y_pred >= t).astype(int), self.termIC, avg=True, returnRuMi=False)
        return np.min(ss)


__all__ = [
    "Method",
    "propagate_go_preds",
    "propagate_ec_preds",
    "normalizedSemanticDistance",
    "normalizedRemainingUncertainty",
    "normalizedMisInformation",
    "bootstrap",
    "load_test_prots",
    "load_go2ic_mapping",
]
