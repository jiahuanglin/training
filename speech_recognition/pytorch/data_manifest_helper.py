import os.path as osp
import os
import csv
import argparse
import sys
import sox



wavfile = '/scratch/jacob/training/speech_recognition/LibriSpeech_dataset/test/wav/2414-128291-0020.wav'
txtfile = '/scratch/jacob/training/speech_recognition/LibriSpeech_dataset/test/txt/2414-128291-0020.txt'


def make_folder(filename):
        temp = osp.dirname(filename)
        if not osp.exists(temp):
                print("Making folder at: {}".format(temp))
                os.makedirs(temp)

def make_file(filename,data=None):
        f = open(filename,"w+")
        f.close()
        if data:
                write_line(filename,data)

def write_line(filename,msg):
        f = open(filename,"a")
        f.write(msg)
        f.close()

def format_entry(entry, root):
        base = osp.basename(entry[0])
        folder = osp.basename(osp.dirname(entry[0]))
        base = base.split('.')[0]+".txt"
        new_file = osp.join(root,folder,base)
        new_entry = entry[2].upper()
        return (new_file, new_entry)

def make_manifest(inputfile, root, idx=-1):
	if idx == -1:
        	idx = ""
	else:
		idx = to_string(idx)
        base = osp.basename(inputfile)
        base = base + idx
        manifest_file = osp.join(root, base)
        make_folder(manifest_file)
        make_file(manifest_file)
        return manifest_file

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', help='some manifest file with the first col containing the wav file path')
	parser.add_argument('--hold_idx', default=-1, type=int)
	parser.add_argument('--stats', dest='stats', action='store_true')
        args = parser.parse_args()
        root = os.getcwd() 			# the root is the current working directory
        filepath = osp.join(os.getcwd(),args.file)
        print("\n\nOpening: {}".format(filepath))
        print("Root: {}".format(root))
	if args.stats:
		manifest_file = filepath + "_stats"
	else: 
        	manifest_file = make_manifest(filepath, root)
        print("Manifest made: {}".format(manifest_file))
        f = open(filepath)
        summary = csv.reader(f,delimiter=',')
        tot = 0
	hold_file = ""
	hold_entry = ""
	for i, row in enumerate(summary):
		tot += 1;
		if args.hold_idx == i:
			(hold_file, hold_entry = format_entry(row, root)
        cur = 0
        f.seek(0)
        for row in summary:
                if cur == 0:
                        cur += 1
                        continue
		if not args.stats:
			if args.hold_idx != -1:
				new_file = hold_file
				new_entry = hold_entry
			else:
	                	(new_file, new_entry) = format_entry(row, root)
                	make_folder(new_file)
                	make_file(new_file, new_entry)
		else:
			seconds = sox.file_info.duration(row[0])
			print(seconds)
			new_file = seconds
                write_line(manifest_file, row[0]+","+new_file+"\n")
                sys.stdout.write("\r[{}/{}]         ".format(cur,tot))
                sys.stdout.flush()
                cur += 1
	sys.stdout.write("\r[{}/{}]         ".format(cur,tot))
        sys.stdout.flush()
        print("\n")

