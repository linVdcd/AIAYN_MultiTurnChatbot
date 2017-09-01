import codecs
from tqdm import tqdm
with codecs.open('train.txt','r','utf-8') as f :
    sentences = f.readlines()

f_p  = codecs.open('corpora/100w.p','w','utf-8')
f_r  = codecs.open('corpora/100w.r','w','utf-8')

for i in tqdm(range(len(sentences))):
    if sentences[i]!='\n' and i+1<len(sentences) and sentences[i+1]!='\n':
        if i-2<0:
            if i-1<0:
                f_p.write('\t\t')
            else:
                f_p.write('\t')
                f_p.write(sentences[i-1].strip('\n')+'\t')
        elif sentences[i-1]!='\n':
            if sentences[i-2]!='\n':
                f_p.write(sentences[i-2].strip('\n')+'\t'+sentences[i-1].strip('\n')+'\t')
            else:f_p.write('\t'+sentences[i-1].strip('\n')+'\t')
        else:f_p.write('\t\t')

        f_p.write(sentences[i])
        f_r.write(sentences[i+1])


f_p.close()
f_r.close()

