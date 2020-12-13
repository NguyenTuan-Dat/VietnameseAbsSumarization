##Delete files
# import glob, os
# os.chdir("/Users/ntdat/Tài liệu/Nghiên Cứu Khoa Học/BertSum/")
# for f in glob.glob("*.story"):
#     os.remove(f)

from vncorenlp import VnCoreNLP
import os

rdrsegmenter = VnCoreNLP("/Users/ntdat/Tài liệu/Nghiên Cứu Khoa Học/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

list_dir = os.listdir("/Users/ntdat/Tài liệu/Nghiên Cứu Khoa Học/DataCrawl_TuoiTre/Details/")
details_dir = "/Users/ntdat/Tài liệu/Nghiên Cứu Khoa Học/DataCrawl_TuoiTre/Details/"
descriptions_dir = "/Users/ntdat/Tài liệu/Nghiên Cứu Khoa Học/DataCrawl_TuoiTre/Descriptions/"
count = 0 
for dir in list_dir:
    try:
        print(dir)
        save_file = open("/Users/ntdat/Tài liệu/Nghiên Cứu Khoa Học/DataCrawl_TuoiTre/RawDataForBertSum/" + str(count) + ".story", "w")
        r_file = open(details_dir+dir, "r")
        lines = r_file.readlines()
        content_body = " ".join(lines)
        sentents = rdrsegmenter.tokenize(content_body)
        content_body = ""
        for s in sentents:
            content_body += " ".join(s) + "\n"

        r_file = open(descriptions_dir+dir, "r")
        lines = r_file.readlines()
        content_summ = ""
        for line in lines:
            if line[-2].isalpha():
                if line[-1] == " ":
                    line[-1] = "."
                else:
                    line += "."
            content_summ += line + " "
        content_summ = content_summ.replace("\n", " ")
        content_summ = content_summ.replace("  ", ".")

        sentents = rdrsegmenter.tokenize(content_summ)
        content_summ = ""
        for s in sentents:
            content_summ += " ".join(s) + "\n"

        content_summ.replace("..", ".")
        sents = content_summ.split(" .")
        content_summ = ""
        for sent in sents:
            if sent == "" or sent == " " or sent == "\n" or sent == ".\n":
                continue
            sent = sent.replace("\n", "")
            if sent[-1] == '.':
                content_summ += "@highlight\n" + sent + "\n"
            else:
                content_summ += "@highlight\n" + sent + " .\n"

        save_file.write(content_body + "\n" + content_summ)
        count+=1
    except Exception:
        continue