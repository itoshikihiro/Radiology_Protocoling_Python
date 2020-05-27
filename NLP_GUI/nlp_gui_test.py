import logging
import re
import sys
import threading
import os

import PySide2
import gensim
import pickle
import pandas as pd

from PySide2.QtGui import QStandardItem
from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,remove_stopwords, stem_text
import numpy as np

from PySide2.QtWidgets import QMainWindow, QApplication, QWidget

from cpt_list import Ui_cpt_list_gui
from icd_list import Ui_icd_list_gui
from nlp_gui import Ui_nlp_gui

# multiple thread class for loading models in different thread in order to save time
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


class cpt_list_gui(QWidget, Ui_cpt_list_gui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setup()

    def setup(self):
        self.cpt_map = {70250: 'radiologic examination, skull; less than four views',
                        71010: 'radiologic examination, chest; single view, frontal',
                        71020: 'radiologic examination, chest, two views, frontal and lateral',
                        71035: 'radiologic examination, chest, special views (eg, lateral decubitus, bucky stu)',
                        71101: 'radiologic exam, ribs, unilateral; inclu posteroanterior chest, min 3 views',
                        72020: 'radiologic examination, spine, single, view, specify level',
                        72040: 'radiologic exam, spine, cervical; two or three views',
                        72050: 'radiologic examination, spine cervical; minimum of four views',
                        72070: 'radiologic exam, spine; thoracic, two views',
                        72072: 'radiologic exam, spine; thoracic, three views',
                        72100: 'radiologic exam, spine, lumbosacral; two or three views',
                        72110: 'radiologic exam, spine, lumbosacral;minimum of four views',
                        72170: 'radiologic examination, pelvis; one or two views',
                        73030: 'radiologic examination, shoulder; complete, minimum of two views',
                        73060: 'radiologic examination; humerus, minimum of two views',
                        73080: 'radiologic exam, elbow; complete minimum of three views',
                        73090: 'radiologic exam; forearm, two views',
                        73110: 'radiologic exam, wrist; complete, minimum of three views',
                        73130: 'radiologic examination, hand; minimum of three views',
                        73140: 'radiologic examination, finger(s), minimum of two views',
                        73502: 'radex hip unilateral with pelvis 2-3 views',
                        73552: 'radiologic examination femur minimum 2 views',
                        73560: 'radiologic exam, knee;one or two views',
                        73562: 'radiologic exam, knee;three views',
                        73590: 'radiologic exam; tibia & fibula, two views',
                        73610: 'radiologic exam, ankle; complete minimum of three views',
                        73630: 'radiologic examination, foot; complete, minimum of three views',
                        74000: 'radiologic examination, abdomen; single anteroposterior view',
                        74020: 'radiologic exam, abdomen; complete incl decubitus &/or erect views',
                        74022: 'radiologic exam abdomen;complete acute abd inc supine,erect &/or decubitus,snglview chest'}
        for cpt in self.cpt_map.keys():
            output_text = "  "+self.cpt_map.get(cpt)
            self.cpt_list.addItem(str(cpt))
            self.cpt_list.addItem(output_text)

class icd_list_gui(QWidget, Ui_icd_list_gui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setup()

    def setup(self):
        self.icd_map = {'A00-B99': 'Certain infectious and parasitic diseases',
                        'C00-D49': 'Neoplasms',
                        'D50-D89': 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
                        'E00-E89': 'Endocrine, nutritional and metabolic diseases',
                        'F01-F99': 'Mental, Behavioral and Neurodevelopmental disorders',
                        'G00-G99': 'Diseases of the nervous system',
                        'H00-H59': 'Diseases of the eye and adnexa',
                        'H60-H95': 'Diseases of the ear and mastoid process',
                        'I00-I99': 'Diseases of the circulatory system',
                        'J00-J99': 'Diseases of the respiratory system',
                        'K00-K95': 'Diseases of the digestive system',
                        'L00-L99': 'Diseases of the skin and subcutaneous tissue',
                        'M00-M99': 'Diseases of the musculoskeletal system and connective tissue',
                        'N00-N99': 'Diseases of the genitourinary system',
                        'O00-O9A': 'Pregnancy, childbirth and the puerperium',
                        'P00-P96': 'Certain conditions originating in the perinatal period',
                        'Q00-Q99': 'Congenital malformations, deformations and chromosomal abnormalities',
                        'R00-R99': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
                        'S00-T88': 'Injury, poisoning and certain other consequences of external causes',
                        'U00-U85': 'Codes for special purposes(U07-U85)',
                        'V00-Y99': 'External causes of morbidity',
                        'Z00-Z99': 'Factors influencing health status and contact with health services'}
        for icd in self.icd_map.keys():
            output_text = "  "+self.icd_map.get(icd)
            self.icd_list.addItem(str(icd))
            self.icd_list.addItem(output_text)




# bind function
# the gui class
class NLP_GUI(QMainWindow, Ui_nlp_gui):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setup()
        # test connect two list gui
        self.cpt_list_gui = cpt_list_gui()
        self.icd_list_gui = icd_list_gui()
        self.cpt_list_gui.show()
        self.icd_list_gui.show()
        self.show()

    # the binding functions
    def setup(self):
        # load all models
        task_1 = MyThread(load_w2v_self_model, ())
        task_2 = MyThread(load_cpt_model, ())
        task_3 = MyThread(load_icd_model, ())
        task_1.start()
        task_2.start()
        task_3.start()
        self.wv = task_1.get_result()
        self.cpt_model = task_2.get_result()
        self.icd_model = task_3.get_result()
        # setup mappings
        # the cpt code mapping
        self.cpt_map = {'70250': 'radiologic examination, skull; less than four views',
                        '71010': 'radiologic examination, chest; single view, frontal',
                        '71020': 'radiologic examination, chest, two views, frontal and lateral',
                        '71035': 'radiologic examination, chest, special views (eg, lateral decubitus, bucky stu)',
                        '71101': 'radiologic exam, ribs, unilateral; inclu posteroanterior chest, min 3 views',
                        '72020': 'radiologic examination, spine, single, view, specify level',
                        '72040': 'radiologic exam, spine, cervical; two or three views',
                        '72050': 'radiologic examination, spine cervical; minimum of four views',
                        '72070': 'radiologic exam, spine; thoracic, two views',
                        '72072': 'radiologic exam, spine; thoracic, three views',
                        '72100': 'radiologic exam, spine, lumbosacral; two or three views',
                        '72110': 'radiologic exam, spine, lumbosacral;minimum of four views',
                        '72170': 'radiologic examination, pelvis; one or two views',
                        '73030': 'radiologic examination, shoulder; complete, minimum of two views',
                        '73060': 'radiologic examination; humerus, minimum of two views',
                        '73080': 'radiologic exam, elbow; complete minimum of three views',
                        '73090': 'radiologic exam; forearm, two views',
                        '73110': 'radiologic exam, wrist; complete, minimum of three views',
                        '73130': 'radiologic examination, hand; minimum of three views',
                        '73140': 'radiologic examination, finger(s), minimum of two views',
                        '73502': 'radex hip unilateral with pelvis 2-3 views',
                        '73552': 'radiologic examination femur minimum 2 views',
                        '73560': 'radiologic exam, knee;one or two views',
                        '73562': 'radiologic exam, knee;three views',
                        '73590': 'radiologic exam; tibia & fibula, two views',
                        '73610': 'radiologic exam, ankle; complete minimum of three views',
                        '73630': 'radiologic examination, foot; complete, minimum of three views',
                        '74000': 'radiologic examination, abdomen; single anteroposterior view',
                        '74020': 'radiologic exam, abdomen; complete incl decubitus &/or erect views',
                        '74022': 'radiologic exam abdomen;complete acute abd inc supine,erect &/or decubitus,snglview chest'}
        # the icd code mapping
        self.icd_array = {0: 'A00-B99',
                          1: 'C00-D49',
                          2: 'D50-D89',
                          3: 'E00-E89',
                          4: 'G00-G99',
                          5: 'I00-I99',
                          6: 'J00-J99',
                          7: 'K00-K95',
                          8: 'L00-L99',
                          9: 'M00-M99',
                          10: 'N00-N99',
                          11: 'R00-R99',
                          12: 'S00-T88',
                          13: 'V00-Y99',
                          14: 'Z00-Z99'}
        self.icd_map = {'A00-B99': 'Certain infectious and parasitic diseases',
                        'C00-D49': 'Neoplasms',
                        'D50-D89': 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
                        'E00-E89': 'Endocrine, nutritional and metabolic diseases',
                        'F01-F99': 'Mental, Behavioral and Neurodevelopmental disorders',
                        'G00-G99': 'Diseases of the nervous system',
                        'H00-H59': 'Diseases of the eye and adnexa',
                        'H60-H95': 'Diseases of the ear and mastoid process',
                        'I00-I99': 'Diseases of the circulatory system',
                        'J00-J99': 'Diseases of the respiratory system',
                        'K00-K95': 'Diseases of the digestive system',
                        'L00-L99': 'Diseases of the skin and subcutaneous tissue',
                        'M00-M99': 'Diseases of the musculoskeletal system and connective tissue',
                        'N00-N99': 'Diseases of the genitourinary system',
                        'O00-O9A': 'Pregnancy, childbirth and the puerperium',
                        'P00-P96': 'Certain conditions originating in the perinatal period',
                        'Q00-Q99': 'Congenital malformations, deformations and chromosomal abnormalities',
                        'R00-R99': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
                        'S00-T88': 'Injury, poisoning and certain other consequences of external causes',
                        'U00-U85': 'Codes for special purposes(U07-U85)',
                        'V00-Y99': 'External causes of morbidity',
                        'Z00-Z99': 'Factors influencing health status and contact with health services'}
        self.cpt_predict_button.clicked.connect(self.cpt_prediction)
        self.icd_predict_button.clicked.connect(self.icd_prediction)
        self.whole_predict_button.clicked.connect(self.whole_text_prediction)
        self.cpt_correct_button.clicked.connect(self.cpt_correct)
        self.icd_correct_button.clicked.connect(self.icd_correct)
        self.whole_cpt_correct_button.clicked.connect(self.whole_cpt_correct)
        self.whole_icd_correct_button.clicked.connect(self.whole_icd_corect)
        # initialize the saved dataframe
        self.cpt_df = pd.DataFrame(columns=['cpt_text', 'cpt_code', 'cpt_code_description'])
        self.cpt_df.loc[0] = None
        self.icd_df = pd.DataFrame(columns=['icd_text', 'icd_code', 'icd_code_description'])
        self.icd_df.loc[0] = None
        self.whole_df = pd.DataFrame(columns=['whole_text', 'cpt_code','cpt_code_description', 'icd_code', 'icd_code_description'])
        self.whole_df.loc[0] = None

        # initialize the saved file names
        self.cpt_file_dir = './saved_records/saved_cpt_records.csv'
        self.icd_file_dir = './saved_records/saved_icd_records.csv'
        self.whole_file_dir = './saved_records/saved_whole_records.csv'

    def cpt_prediction(self):
        if os.path.isfile(self.cpt_file_dir):
            self.cpt_df.to_csv(self.cpt_file_dir, mode='a', index=False, header=False)
        else:
            self.cpt_df.to_csv(self.cpt_file_dir, mode='w', index=False)
        ori_cpt_text = str(self.cpt_input_text.toPlainText())
        cpt_text = clean_txt(ori_cpt_text)
        cpt_text = word_averaging(self.wv, cpt_text)
        cpt_text = cpt_text.reshape([1, 300])
        result = self.cpt_model.predict(cpt_text)
        cpt_code = str(result[0])
        cpt_description = self.cpt_map.get(cpt_code)
        output_text = cpt_code +'\n\n'+ cpt_description
        self.cpt_output_text.setPlainText(output_text)
        self.cpt_df.loc[0] = [ori_cpt_text, cpt_code, cpt_description]

    def cpt_correct(self):
        changed_cpt_code = str(self.cpt_correct_text.toPlainText())
        changed_cpt_desci = self.cpt_map.get(changed_cpt_code)
        ori_cpt_text = self.cpt_df.loc[0][0]
        self.cpt_df.loc[0] = [ori_cpt_text, changed_cpt_code, changed_cpt_desci]

    def icd_prediction(self):
        if os.path.isfile(self.icd_file_dir):
            self.icd_df.to_csv(self.icd_file_dir, mode='a', index=False, header=False)
        else:
            self.icd_df.to_csv(self.icd_file_dir, mode='w', index=False)
        ori_icd_text = str(self.icd_input_text.toPlainText())
        icd_text = clean_txt(ori_icd_text)
        icd_text = word_averaging(self.wv, icd_text)
        icd_text = icd_text.reshape([1, 300])
        result = self.icd_model.predict(icd_text)
        icd_code = str(self.icd_array.get(result[0]))
        icd_descri = self.icd_map.get(icd_code)
        output_text = icd_code +'\n\n'+ icd_descri
        self.icd_output_text.setPlainText(output_text)
        self.icd_df.loc[0] = [ori_icd_text, icd_code, icd_descri]

    def icd_correct(self):
        changed_icd_code = str(self.icd_correct_text.toPlainText())
        changed_icd_desci = self.icd_map.get(changed_icd_code)
        ori_icd_text = self.icd_df.loc[0][0]
        self.icd_df.loc[0] = [ori_icd_text, changed_icd_code, changed_icd_desci]

    def whole_text_prediction(self):
        if os.path.isfile(self.whole_file_dir):
            self.whole_df.to_csv(self.whole_file_dir, mode='a', index=False, header=False)
        else:
            self.whole_df.to_csv(self.whole_file_dir, mode='w', index=False)
        whole_text = str(self.whole_input_text.toPlainText())
        # cpt part
        cpt_text = remove_unreadable(whole_text)
        cpt_text = cpt_format_str(cpt_text)
        cpt_text = remove_special_char(cpt_text)
        cpt_text = cpt_ext(cpt_text)
        cpt_text = clean_txt(cpt_text)
        cpt_text = word_averaging(self.wv, cpt_text)
        cpt_text = cpt_text.reshape([1, 300])
        cpt_result = self.cpt_model.predict(cpt_text)

        # icd part
        icd_text = remove_unreadable(whole_text)
        icd_text = icd_format_str(icd_text)
        icd_text = remove_special_char(icd_text)
        icd_text = icd_enhance_formatting(icd_text)
        icd_text = icd_ext(icd_text)
        icd_text = clean_txt(icd_text)
        icd_text = word_averaging(self.wv, icd_text)
        icd_text = icd_text.reshape([1, 300])
        icd_result = self.icd_model.predict(icd_text)

        # show result part
        cpt_code = str(cpt_result[0])
        cpt_description = self.cpt_map.get(cpt_code)
        cpt_output_text = cpt_code +'\n\n'+ cpt_description
        icd_code = str(self.icd_array.get(icd_result[0]))
        icd_description = self.icd_map.get(icd_code)
        icd_output_text = icd_code + '\n\n' + icd_description
        self.whole_cpt_output_text.setPlainText(cpt_output_text)
        self.whole_icd_output_text.setPlainText(icd_output_text)
        self.whole_df.loc[0] = [whole_text, cpt_code, cpt_description, icd_code, icd_description]

    def whole_cpt_correct(self):
        changed_cpt_code = str(self.whole_cpt_correct_text.toPlainText())
        changed_cpt_desci = self.cpt_map.get(changed_cpt_code)
        ori_whole_text = self.whole_df.loc[0][0]
        ori_icd_code = self.whole_df.loc[0][3]
        ori_icd_des = self.whole_df.loc[0][4]
        self.whole_df.loc[0] = [ori_whole_text, changed_cpt_code, changed_cpt_desci, ori_icd_code, ori_icd_des]

    def whole_icd_corect(self):
        changed_icd_code = str(self.whole_icd_correct_text.toPlainText())
        changed_icd_desci = self.icd_map.get(changed_icd_code)
        ori_whole_text = self.whole_df.loc[0][0]
        ori_cpt_code = self.whole_df.loc[0][1]
        ori_cpt_des = self.whole_df.loc[0][2]
        self.whole_df.loc[0] = [ori_whole_text, ori_cpt_code, ori_cpt_des, changed_icd_code, changed_icd_desci]

    # rewrite the actions before the main window closing
    def closeEvent(self, event:PySide2.QtGui.QCloseEvent):
        # record the last prediction of CPT code
        if not self.cpt_df.loc[0].isnull().all():
            if os.path.isfile(self.cpt_file_dir):
                self.cpt_df.to_csv(self.cpt_file_dir, mode='a', index=False, header=False)
            else:
                self.cpt_df.to_csv(self.cpt_file_dir, mode='w', index=False)
        # record the last prediction of ICD code
        if not self.icd_df.loc[0].isnull().all():
            if os.path.isfile(self.icd_file_dir):
                self.icd_df.to_csv(self.icd_file_dir, mode='a', index=False, header=False)
            else:
                self.icd_df.to_csv(self.icd_file_dir, mode='w', index=False)
        # record the last prediction of whole text
        if not self.whole_df.loc[0].isnull().all():
            if os.path.isfile(self.whole_file_dir):
                self.whole_df.to_csv(self.whole_file_dir, mode='a', index=False, header=False)
            else:
                self.whole_df.to_csv(self.whole_file_dir, mode='w', index=False)

        print("the main window was closed")
        self.cpt_list_gui.close()
        self.icd_list_gui.close()


def load_cpt_model():
    print("loading cpt model")
    loaded_model = pickle.load(open('./models/cpt_w2v_self_xgboost.sav', 'rb'))
    return loaded_model

def load_icd_model():
    print("loading icd model")
    loaded_model = pickle.load(open('./models/icd_w2v_self_xgboost.sav', 'rb'))
    return loaded_model

def load_w2v_self_model():
    print("loading w2v self model")
    wv = gensim.models.KeyedVectors.load_word2vec_format("./models/embedding_word2vec.txt", binary=False)
    wv.init_sims(replace=True)
    return wv

def remove_special_char(txt):
    return re.sub(r'[^a-zA-Z0-9 :,_/;.]', r'', txt)


def clean_txt(txt):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,remove_stopwords, stem_text]
    words = preprocessing.preprocess_string(txt.lower(), CUSTOM_FILTERS)
    new_words = []
    for word in words:
        word_val = word
        word_val = re.sub(r'([a-zA-Z])+\d+([a-zA-Z])+',r' ', word_val)
        word_val = re.sub(r'([a-zA-Z])+\d+',r' ', word_val)
        word_val = re.sub(r'\d+([a-zA-Z])+',r' ', word_val)
        if word_val == " ":
            new_words = []
            break;
        new_words.append(word)
    if not new_words:
        return 'N/A'
    return new_words

def remove_unreadable(txt):
    return re.sub(r'_[a-zA-Z0-9]+_',r'\n',txt)

def format_str(txt):
    return_val = txt.replace('\r',' ')
    return_val = return_val.strip()
    return_val = re.sub(r'(\s*\n\s*){2,}',r';', return_val)
    return_val = return_val.replace('(\n)+',' ')
    return_val = re.sub(r'(\s)+',r' ', return_val)
    return_val = return_val.strip()
    return return_val

def cpt_ext(txt):
    try:
        splited_list = txt.split(';')
        new_txt = ""
        for i in splited_list:
            for j in i.split(':'):
                if j.strip().lower()=='procedure':
                    new_txt=i.split(':')[1].strip()
                    return new_txt
                elif j.strip().lower()=='exam':
                    new_txt=i.split(':')[1].strip()
                    return new_txt
        for i in range(len(splited_list)):
            tmp_txt = splited_list[i].strip()
            if 'REPORT'== tmp_txt.split(" ")[0].strip():
                new_txt = tmp_txt.split(" ")[1:]
                new_txt = " ".join(new_txt).strip()
                if new_txt == "":
                    new_txt = splited_list[i+1].strip()
        return new_txt
    except:
        return 'N/A'


def icd_format_str(txt):
    return_val = txt.replace('\r',' ')
    return_val = return_val.strip()
    return_val = re.sub(r'(\s*\n\s*){2,}',r';;;', return_val)
    return_val = return_val.replace('(\n)+',' ')
    return_val = re.sub(r'(\s)+',r' ', return_val)
    return_val = return_val.strip()
    return return_val


def icd_enhance_formatting(txt):
    new_str = ""
    for tmp_str in txt.split(';;;'):
        if ":" in tmp_str:
            new_str += ";;;"
        new_str += tmp_str.strip() +" "
    return new_str

def icd_ext(txt):
    try:
        splited_list = txt.split(';;;')
        new_txt = ""
        for i in splited_list:
            if ':' in i:
                i_split_list = i.split(':')
                prefix = i_split_list[0].lower()
                if 'impression' in prefix:
                    new_txt += i_split_list[1].lower() + '; '
                if 'history' in prefix:
                    new_txt += i_split_list[1].lower() + '; '
                if 'indication' in prefix:
                    new_txt += i_split_list[1].lower() + '; '
        return new_txt.strip()
    except:
        return 'N/A'

def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.vectors_norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        print(words)
        return np.zeros(wv.vector_size,)

    mean = np.array(mean).mean(axis=0)
    return mean

def cpt_format_str(txt):
    return_val = txt.replace('\r',' ')
    return_val = return_val.strip()
    return_val = re.sub(r'(\s*\n\s*){2,}',r';', return_val)
    return_val = return_val.replace('(\n)+',' ')
    return_val = re.sub(r'(\s)+',r' ', return_val)
    return_val = return_val.strip()
    return return_val

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # cpt_app_gui = cpt_list_gui()
    # icd_app_gui = icd_list_gui()
    nlp_gui = NLP_GUI()
    sys.exit(app.exec_())
