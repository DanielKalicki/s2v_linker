import os.path
import json
import bz2
import pickle
import random
import re
import copy
import math
from urllib.parse import unquote
import numpy as np
import torch
from torch.utils.data import Dataset
from flair.embeddings import RoBERTaEmbeddings
from flair.data import Sentence
# from s2v_embedding_models.s2v_gammaTransformer.s2v_gammaTransformer import generate_s2v

random.seed(0)
batch_train_data = []
batch_valid_data = []
batch_full_data = {}
links_dict = {}


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def find_html_links(text):
    links = []
    search_results = re.findall('<a.*?>', text)
    for link in search_results:
        links.append(unquote(
            link.replace("<a href=\"", "").replace("\">", "")).lower())
    return links


def find_html_links_from_wikipage(text):
    links = []
    # search_results = re.findall('\shref.*?>', text)
    search_results = re.findall("href=\"view-source:https://en.wikipedia.org/wiki/.*?>", text)
    search_results = re.findall("<a href=\"https://en.wikipedia.org/wiki/.*?\s", text)
    for link in search_results:
        links.append(unquote(
            link.replace("<a href=\"https://en.wikipedia.org/wiki/", "")[:-2]))
    return links


def create_linkset(input_articles):
    article_dict = {}
    datasets_dir = "./datasets/hotpotqa/"
    for folder in sorted(os.listdir(datasets_dir)):
        for file in sorted(os.listdir(datasets_dir+folder)):
            print(datasets_dir+folder+'/'+file)
            if file.split(".")[-1] == 'bz2':
                with bz2.BZ2File(datasets_dir+folder+"/"+file, "r") as fp:
                    for line in fp:
                        data = json.loads(line)
                        title = data['title'].lower()
                        text = data['text']
                        if title in input_articles:
                            links = {}
                            for p_idx, paragraph in enumerate(text[1:]): # skip the first line with title
                                links[p_idx+1] = {}
                                for l_idx, line_ in enumerate(paragraph):
                                    links[p_idx+1][l_idx] = find_html_links(line_)
                            article_dict[title] = {
                                'links': links
                            }
    with open('./train/_datasets_1s/full_art_links.json', 'w') as fw:
        fw.write(json.dumps(article_dict, sort_keys=True, indent=4))
    exit(1)

# create_linkset()


def load_linkset():
    with open('./train/_datasets_1s/links.json', 'r') as f:
        linkset = json.load(f)
        return linkset


def get_links(title, linkset):
    try:
        return linkset[title]['links']
    except KeyError:
        return []


def find_all_links(title, linkset, depth=100):
    links = set()
    links.update(get_links(title, linkset))
    links_length = len(links)
    for _ in range(0, depth):
        for link in copy.copy(links):
            links.update(get_links(link, linkset))
        if len(links) == links_length:
            break
        else:
            links_length = len(links)
    return links


class WikiLinksBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config
        self.valid = valid

        self.datasets_dir = "./datasets/hotpotqa/"
        self.batch_dir = './train/_datasets_1s/'

        # input_articles = self._get_input_articles_list_from_vital_articles()
        # create_linkset(input_articles)
        # linkset = load_linkset()
        # all_articles = self._get_articles_to_emb(linkset)
        # self._create_batch_words_emb(all_articles, linkset)
        # self._create_batch_s2v()
        # exit(1)

        self.train_batch_part = -1
        self._init_batch()

    def _get_input_articles_list_from_vital_articles(self):
        self.vital_articles_dump_dir = './datasets/wiki/'
        articles = set()
        for file in sorted(os.listdir(self.vital_articles_dump_dir)):
            if file.split(".")[-1] == 'html':
                cnt = 0
                if 'html' in file:
                    with open(self.vital_articles_dump_dir+file, "r") as f:
                        for link in find_html_links_from_wikipage(f.read()):
                            key_words_list = [":", "Main_Page"]
                            if all(word not in link for word in key_words_list):
                                article = link.replace("_", " ")
                                articles.update([article.lower()])
                                cnt += 1
                        # print(file)
                        # break
        print('Number of input articles: '+ str(len(articles)))
        return articles
    
    def _get_articles_to_emb(self, linkset):
        all_articles = set()
        for article in linkset:
            links = linkset[article]['links']
            if len(links) > 0:
                # all_articles.update([article])
                for link in links:
                    if link.lower() in linkset:
                        all_articles.update([link.lower()])
        print('Number of all articles: '+ str(len(all_articles)))
        return all_articles

    def _embed_first_sentence(self, text):
        first_sentence = text[1][0]
        sentence = remove_html_tags(first_sentence)
        sentence_emb = self._process_sentences([sentence])
        return {
            'first_sentence_emb': sentence_emb
        }

    def _create_batch_words_emb(self, all_articles, linkset):
        self.embedding = RoBERTaEmbeddings(
            pretrained_model_name_or_path="roberta-large",
            layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20," +
                   "21,22,23,24",
            pooling_operation="mean", use_scalar_mix=True)
        cnt = 0
        all_articles_lc = [x.lower() for x in all_articles]
        for folder in sorted(os.listdir(self.datasets_dir)):
            if not os.path.isfile(self.batch_dir + folder + "_" + 'articles_wEmb.pickle'):
                article_dict = {}
                for file in sorted(os.listdir(self.datasets_dir+folder)):
                    if file.split(".")[-1] == 'bz2':
                        print(self.datasets_dir+folder+'/'+file)
                        with bz2.BZ2File(self.datasets_dir+folder+"/"+file, "r") as fp:
                            for line in fp:
                                data = json.loads(line)
                                title = data['title'].lower()
                                text = data['text']
                                if title in all_articles_lc:
                                    # embed only first sentece because this s2v will be used only as an training label
                                    try:
                                        article_dict[title] = self._embed_first_sentence(text)
                                        article_dict[title]['links'] = []
                                        for link in linkset[title]['links']:
                                            article_dict[title]['links'].append(link.lower())
                                    except IndexError:
                                        print('\tIndexError')
                                cnt += 1
                pickle.dump(article_dict, open(self.batch_dir + folder + "_" + 'articles_wEmb.pickle', 'wb'))

    def _process_sentences(self, sentences):
        sentences_emb = []
        for sentence in sentences:
            sentence = " ".join(sentence.split())
            sent = sentence
            if len(sent.strip()) == 0:
                sent = 'empty'
            try:
                sent = Sentence(sent)
                self.embedding.embed(sent)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
            except IndexError:
                print('IndexError')
                print(sentence)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
        sentences_emb_short = sentences_emb
        return sentences_emb_short

    def _create_batch_s2v(self):
        for file in sorted(os.listdir(self.batch_dir)):
            if (file.split(".")[-1] == 'pickle') and (not os.path.isfile(self.batch_dir + file.replace('_wEmb.', '_s2v.'))) and \
               ('_wEmb' in file):
                print(file)
                data = pickle.load(open(self.batch_dir+file, 'rb'))
                for title in data:
                    s2v_gammaTransformer = self._generate_s2v(data[title]['first_sentence_emb'])
                    del data[title]['first_sentence_emb']
                    data[title]['first_sentence_vect_gTr'] = s2v_gammaTransformer[0]
                pickle.dump(data, open(self.batch_dir + file.replace('_wEmb.', '_s2v.'), 'wb'))

    def _generate_s2v(self, sentence):
        sent1_s2v, _, _ = generate_s2v(sentence)
        return sent1_s2v

    def _init_batch(self):
        global batch_train_data, batch_valid_data, batch_full_data, links_dict
        if not self.valid:
            articles_inputs = {}
            for file in sorted(os.listdir(self.batch_dir)):
                if (file.split(".")[-1] == 'pickle') and ('s2v' in file):
                    data = pickle.load(open(self.batch_dir+file, 'rb'))
                    articles_inputs.update(data)
            for title in articles_inputs:
                sentence_emb = articles_inputs[title]['first_sentence_vect_gTr']
                links = articles_inputs[title]['links']
                batch_data = {
                    'sentence_vect_gTr': sentence_emb,
                    'links': links
                }
                links_ok = False
                for link in links:
                    if link in articles_inputs:
                        links_ok = True
                        if not link.lower() in links_dict:
                            links_dict[link.lower()] = 0
                        links_dict[link.lower()] += 1
                if links_ok:
                    batch_train_data.append(title)
                batch_full_data[title] = batch_data
            random.shuffle(batch_train_data)
            batch_valid_data = batch_train_data[:int(len(batch_train_data)*0.1)]
            batch_train_data = batch_train_data[int(len(batch_train_data)*0.1):]
            print(batch_valid_data[0:10])

            # del articles_inputs
            print("Train dataset size: " + str(len(batch_train_data)))
            print("Test dataset size: " + str(len(batch_valid_data)))

    def _find_closest_s2v(self, s2v):
        sim_dict = {}
        a = s2v
        for art in batch_full_data:
            b = batch_full_data[art]['sentence_vect_gTr']
            sim = ((a - b)**2).mean(axis=0)
            if sim not in sim_dict:
                sim_dict[sim] = []
            sim_dict[sim].append(art)
        for key in sorted(sim_dict)[0:11]:
            print('\t'+str(key)+'\t'+str(sim_dict[key]))

    def on_epoch_end(self):
        self._init_batch()

    def __len__(self):
        global batch_train_data, batch_valid_data
        if self.valid:
            return len(batch_valid_data)
        else:
            return len(batch_train_data)

    def __getitem__(self, idx):
        global batch_train_data, batch_valid_data, links_dict
        batch_dataset = batch_valid_data if self.valid else batch_train_data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        while True:
            if not self.valid:
                idx = random.randint(0, len(batch_dataset)-1)
            num_of_links = 3
            input_sentence = torch.zeros((self.config['s2v_dim'],),
                                        dtype=torch.float)
            linked_sentence = torch.rand((num_of_links, self.config['s2v_dim'],),
                                        dtype=torch.float) * 100
            link_mask = torch.zeros((num_of_links, self.config['s2v_dim']), dtype=torch.float)
            loss_threshold = torch.zeros((1), dtype=torch.float)

            input_title = batch_dataset[idx]
            input_sentence = torch.from_numpy(
                batch_full_data[input_title]['sentence_vect_gTr'].astype(np.float32))
            links = batch_full_data[input_title]['links']
            link_cnt_sum = 1
            cnt = 0
            valid_links = []
            for i in range(len(links)):
                try:
                    linked_art = links[i].lower()
                    linked_sentence[cnt] = torch.from_numpy(
                        batch_full_data[linked_art]['sentence_vect_gTr']
                            .astype(np.float32))
                    link_mask[cnt] = torch.from_numpy(np.ones((self.config['s2v_dim'],))
                                                        .astype(np.float32))
                    link_cnt_sum += links_dict[links[i].lower()]
                    valid_links.append(linked_art)
                    cnt += 1
                    if cnt >= num_of_links:
                        break
                except KeyError:
                    pass
            # rnd = random.randint(0, int(math.log2(link_cnt_sum)))
            # if (rnd == 0) or (self.valid):
            #     break
            if self.valid:
                loss_threshold = torch.tensor(0.0)
            else:
                loss_threshold = torch.tensor(math.log10(link_cnt_sum)*0.0025)
            # print(input_title)
            # print(valid_links)
            break
        # print(link_mask)
        return (input_sentence, linked_sentence, link_mask, loss_threshold)

    def get_discriminator_batch(self, generated_batch):
        global batch_links_data

        # print(generated_batch)
        batch_size = len(generated_batch)
        # print(batch_size)

        batch = torch.zeros((batch_size, self.config['s2v_dim']),
                            dtype=torch.float)
        labels = torch.zeros((batch_size, 2), dtype=torch.float)

        for i_batch in range(batch_size):
            # True - real data
            # False - generated data
            rnd = random.choice([True, False])
            labels[i_batch][int(rnd)] = 1.0

            if rnd:
                rnd_art_idx = random.randint(0, len(batch_links_data)-1)
                rnd_art_title = list(batch_links_data.keys())[rnd_art_idx]
                batch[i_batch] = torch.from_numpy(
                    batch_links_data[rnd_art_title]['first_sentence_vect_gTr']
                        .astype(np.float32))
            else:
                batch[i_batch] = generated_batch[i_batch]


        return (batch, labels)

    def check_accuracy(self, prediction):
        self._find_closest_s2v(prediction)


def test():
    batcher = WikiLinksBatch({
        's2v_dim': 4096
    })
    for i in range(100):
        batcher.__getitem__(i)
    # print(x)
    # print(y)


# test()
