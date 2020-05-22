import numpy as np
from numpy.linalg import norm
import os.path
import json
import bz2
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import random
import re
from urllib.parse import unquote
import torch
random.seed(0)
from flair.embeddings import RoBERTaEmbeddings
from flair.data import Sentence
import copy
# from s2v_embedding_models.s2v_gammaTransformer.s2v_gammaTransformer import generate_s2v

batch_train_data = {}
batch_links_data = {}
batch_valid_data = {}


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def find_html_links(text):
    links = []
    search_results = re.findall('<a.*?>', text)
    for link in search_results:
        links.append(unquote(
            link.replace("<a href=\"", "").replace("\">", "")))
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


def create_linkset():
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
                        try:
                            first_sentence = text[1][0]
                            links = find_html_links(first_sentence)
                            article_dict[title] = {
                                'links': links
                            }
                        except IndexError:
                            pass
    with open('./train/datasets/links.json', 'w') as fw:
        fw.write(json.dumps(article_dict, sort_keys=True, indent=4))
    exit(1)


# create_linkset()


def load_linkset():
    with open('./train/datasets/links.json', 'r') as f:
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
    for i in range(0, depth):
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
        self.batch_dir = './train/datasets/'

        # input_articles = self._get_input_articles_list_from_vital_articles()
        # linkset = load_linkset()
        # all_articles = self._get_articles_to_emb(input_articles, linkset)
        # self._embed_articles(input_articles, all_articles, linkset)
        # self._embed_articles_s2v()
        # exit(1)

        self.train_batch_part = -1
        self._init_batch()

    def _get_input_articles_list_from_vital_articles(self):
        self.vital_articles_dump_dir = './datasets/wiki/'
        articles = set()
        for file in sorted(os.listdir(self.vital_articles_dump_dir)):
            if file.split(".")[-1] == 'html':
                cnt = 0
                with open(self.vital_articles_dump_dir+file, "r") as f:
                    for link in find_html_links_from_wikipage(f.read()):
                        key_words_list = [":", "Main_Page"]
                        if all(word not in link for word in key_words_list):
                            article = link.replace("_", " ")
                            articles.update([article.lower()])
                            cnt += 1
                pass
        return articles

    def _get_articles_to_emb(self, initial_articles, linkset):
        all_articles = set()
        for article in initial_articles:
            if len(get_links(article, linkset)) > 0:
                all_articles.update([article])
        for title in copy.copy(all_articles):
            links = find_all_links(title, linkset, 0)
            all_articles.update(links)
        return all_articles

    def _embed_articles(self, input_articles, all_articles, linkset):
        self.embedding = RoBERTaEmbeddings(
            pretrained_model_name_or_path="roberta-large",
            layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20," +
                   "21,22,23,24",
            pooling_operation="mean", use_scalar_mix=True)
        article_dict = {}
        cnt = 0
        all_articles_lc = [x.lower() for x in all_articles]
        input_articles_lc = [x.lower() for x in input_articles]
        for folder in sorted(os.listdir(self.datasets_dir)):
            for file in sorted(os.listdir(self.datasets_dir+folder)):
                print(self.datasets_dir+folder+'/'+file)
                if file.split(".")[-1] == 'bz2':
                    with bz2.BZ2File(self.datasets_dir+folder+"/"+file, "r") as fp:
                        for line in fp:
                            data = json.loads(line)
                            title = data['title'].lower()
                            if title in all_articles_lc:
                                text = data['text']
                                try:
                                    first_sentence = text[1][0]
                                    sentence = remove_html_tags(first_sentence)
                                    sentence_emb = self._process_sentences([sentence])
                                    sentence_vect = np.mean(sentence_emb[0], axis=0)
                                    # sentence_vect = self._generate_s2v(sentence_emb)
                                    input_example = False
                                    links = get_links(title, linkset)
                                    if title in input_articles_lc:
                                        input_example = True
                                    if len(links) > 0 or not input_example:
                                        article_dict[title] = {
                                            'first_sentence_vect_mean_pool': sentence_vect,
                                            'first_sentence_emb': sentence_emb,
                                            'links': links,
                                            'input_example': input_example
                                        }
                                        print(title)
                                        print('\t'+str(input_example))
                                        print('\t'+str(get_links(title, linkset)))
                                        print('\t'+str(cnt) + "/" + str(len(all_articles)))
                                except IndexError:
                                    print(text)
                                cnt += 1
            pickle.dump(article_dict, open(self.batch_dir + folder + "_" + 'articles.pickle',
                                           'wb'))
            article_dict = {}


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

    def _embed_articles_s2v(self):
        for file in sorted(os.listdir(self.batch_dir)):
            if file.split(".")[-1] == 'pickle':
                data = pickle.load(open(self.batch_dir+file, 'rb'))
                for title in data:
                    print(title)
                    s2v_gammaTransformer = self._generate_s2v(data[title]['first_sentence_emb'])
                    del data[title]['first_sentence_emb']
                    # print(data[title])
                    # print(len(s2v_gammaTransformer))
                    # print(len(s2v_gammaTransformer[0]))
                    # print(s2v_gammaTransformer[0])
                    data[title]['first_sentence_vect_gTr'] = s2v_gammaTransformer[0]
                    # print(data[title])
            pickle.dump(data, open(self.batch_dir + file.replace('.', '_s2v.'), 'wb'))
            # exit(1)

    def _generate_s2v(self, sentence):
        # should be an array of [sent1_emb, sent2_emb]
        # TODO check
        # print('sentence')
        # print(sentence)
        sent1_s2v, sent2_s2v, prediction = generate_s2v(sentence)
        # print("-----------++++------------")
        # print(sent1_s2v)
        # print("------------------------")
        # exit(1)
        return sent1_s2v

    def _init_batch(self):
        global batch_train_data, batch_valid_data
        if not self.valid:
            articles_dict = {}
            for file in sorted(os.listdir(self.batch_dir)):
                if (file.split(".")[-1] == 'pickle') and ('s2v' in file):
                    data = pickle.load(open(self.batch_dir+file, 'rb'))
                    articles_dict.update(data)
                # if len(articles_dict) > 1000:
                    # break
            all_articles_lc = [x.lower() for x in articles_dict]
            valid_cnt = 0
            for title in articles_dict:
                batch = articles_dict[title]
                if batch['input_example']:
                    link_exist = False
                    for link in batch['links']:
                        if link.lower() in all_articles_lc:
                            link_exist = True
                    if link_exist:
                        # if valid_cnt < 200:
                        if valid_cnt < 2000:
                            batch_valid_data[title.lower()] = batch
                        else:
                            batch_train_data[title.lower()] = batch
                        valid_cnt += 1
                batch_links_data[title.lower()] = batch
            print("Train dataset size: " + str(len(batch_train_data)))
            print("Test dataset size: " + str(len(batch_valid_data)))
            print("Articles count: " + str(len(batch_links_data)))

    def _find_closses_s2v(self, s2v):
        sim_dict = {}
        a = s2v
        for art in batch_links_data:
            # b = batch_links_data[art]['first_sentence_vect_mean_pool']
            b = batch_links_data[art]['first_sentence_vect_gTr']
            # sim = np.dot(a, b)/(norm(a)*norm(b))
            # sim = np.linalg.norm(a-b)
            sim = ((a - b)**2).mean(axis=0)
            if sim not in sim_dict:
                sim_dict[sim] = []
            sim_dict[sim].append(art)
        for key in sorted(sim_dict)[1:3]:
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
        global batch_train_data, batch_valid_data
        batch_dataset = batch_valid_data if self.valid else batch_train_data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_sentence = torch.zeros((self.config['s2v_dim'],),
                                     dtype=torch.float)
        linked_sentence = torch.rand((12, self.config['s2v_dim'],),
                                      dtype=torch.float) * 100

        art_title = list(batch_dataset.keys())[idx]
        input_sentence = torch.from_numpy(
            batch_dataset[art_title]['first_sentence_vect_gTr'].astype(np.float32))
        links = batch_dataset[art_title]['links']
        # random.shuffle(links)
        cnt = 0
        for i in range(len(links)):
            try:
                linked_art = links[i].lower()
                linked_sentence[cnt] = torch.from_numpy(
                    batch_links_data[linked_art]['first_sentence_vect_gTr']
                        .astype(np.float32))
                cnt += 1
                break
            except KeyError:
                pass

        # print(art_title)
        # self._find_closses_s2v(batch_dataset[art_title]['first_sentence_vect_gTr'])
        return (input_sentence, linked_sentence)

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


def test():
    batcher = WikiLinksBatch({
        's2v_dim': 4096
    })
    for i in range(10):
        x, y = batcher.__getitem__(i)
    # print(x)
    # print(y)


# test()
