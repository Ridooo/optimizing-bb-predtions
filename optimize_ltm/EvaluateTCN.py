
from keras import backend as bknd
from keras.callbacks import *
import numpy as np
import logging



def ctc_lambda_func(args):
    iy_pred, ilabels, ilabel_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    
    # create the input length
    bn_shape = iy_pred.get_shape()
    iinput_length = bknd.ones_like(ilabel_length)*int(bn_shape[1])
    
    return bknd.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)


class Evaluate(Callback):
    
    def sets(self, model, generator, label_len, db):
        self.model_eval = model
        self.generator = generator
        self.label_len  = label_len
        self.db = db
        
    def on_epoch_end(self, epoch, logs=None):
        ler, results = evaluate(self.model_eval, self.generator, self.label_len,self.db)
        print('')
        print('ler:'+str(ler))

def evaluate(input_model, generator, label_len, db):
    correct_prediction = 0
    #generator = img_gen_val()

    #[x_test, y_test,_],_ = next(generator)
    [images, heights, widths, lines,y_test,seq_lens],_ =  next(generator)
    # print(" ")
    y_pred = input_model.predict([images, heights, widths, lines]) 
    shape = y_pred[:, 2:, :].shape 
    ctc_decode = bknd.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
    out = bknd.get_value(ctc_decode)[:, :label_len]
    
    ler, results = compare(out, y_test, db.Ivoc, show=len(out)-1)
    
    return ler, results

def compare1(out, y_true, Ivoc, show=-1):
    
    
            
            if len(out)!=2:
                raise Exception("the length of out list must be 2", len(out))
            
            ys_pred = []
            ys_true = []
            
            for m in range(len(out)):
                result_str = ''.join([Ivoc[k] for k in out[m] if k in Ivoc])
                y_testm = ''.join([Ivoc[k] for k in y_true[m] if k in Ivoc])
                if len(y_testm)==0:
                    y_testm = ''
                ys_pred.append(result_str)
                ys_true.append(y_testm)
            ys_pred = ' '.join(ys_pred)
            ys_true = ' '.join(ys_true)
            
            logging.debug("before stripping and replacing %s , %s, %s, %s, %s", ys_pred==ys_true,'P:',ys_pred,', T:',ys_true )
            ys_pred = ys_pred.strip().replace('  ',' ')
            ys_true = ys_true.strip().replace('  ',' ')
                
            logging.debug("after stripping and replacing %s , %s, %s, %s, %s", ys_pred==ys_true,'P:',ys_pred,', T:',ys_true )
                
            ler = levenshtein(ys_pred, ys_true)
            
            logging.debug("length ys_true %f",len(ys_true))
            logging.debug("length ys_pred %f",len(ys_pred))
                
            ler = ler/(len(ys_true)+0.00000001)# if len(y_testm)>0 else lerI
            logging.debug("ler %f", ler)
                #result_str = result_str.replace('-', '')
            #results.append([result_str, lerI])
            return ler
        
def compare(out, y_test, Ivoc, show=-1):
            ler = 0
            results = []
            
            
            for m in range(len(out)):
                result_str = ''.join([Ivoc[k] for k in out[m] if k in Ivoc])
                y_testm = ''.join([Ivoc[k] for k in y_test[m] if k in Ivoc])
                if len(y_testm)==0:
                    y_testm = ''
                #logging.debug("%i before stripping and replacing %s , %s, %s, %s, %s",m, result_str==y_testm,'P:',result_str,', T:',y_testm )
                result_str = result_str.strip().replace('  ',' ')
                y_testm = y_testm.strip().replace('  ',' ')
                #if show>m:
                        #logging.debug("%i after stripping and replacing %s , %s, %s, %s, %s", m,result_str==y_testm,'P:',result_str,', T:',y_testm )
                
                lerI = levenshtein(result_str, y_testm)
                #logging.debug("%i lerI %f",lerI)
                #logging.debug("%i len y_testm %f",len(y_testm))
                #logging.debug("%i len result_str %f",len(result_str))
                
                ler += lerI/(len(y_testm)+0.00000001)# if len(y_testm)>0 else lerI
                #logging.debug("%i ler %f",m, ler)
                #result_str = result_str.replace('-', '')
                results.append([result_str, lerI])
            return ler/len(out), results
        
def getPrediction(model_eval, x_test):    
            label_len = 1000
            y_pred = model_eval.predict(x_test, batch_size=15) 
            shape = y_pred[:, 2:, :].shape 
            ctc_decode = bknd.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
            out = bknd.get_value(ctc_decode)[:, :label_len]
            return out


def levenshtein(source, target, mat = None):
    #todos:\/
    #add different score for changing a letter into another
    #add different score for adding a letter before and after another letter ?
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))
    
    mm = np.ones((len(source),len(target)))
    #mat = getMat()
    if mat is not None:
        for i,p1 in enumerate(source):
            for j,p2 in enumerate(target):
                mm[i,j] = mat[p1,p2]

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1).astype("float32")
    for i,s in enumerate(source):
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], mm[i,:]*(target != s)))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 2)

        previous_row = current_row

    return previous_row[-1]
    