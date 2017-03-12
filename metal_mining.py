import urllib2
import os
import urllib
from bs4 import BeautifulSoup
import numpy as np
import scipy
import scipy.misc
import re
import json
import sys
import h5py

# metaldata.txt should be like
# Anthrax,0,Thrash 0 means all
# Sepultura,1-4,Thrash,5-,Groove

def align_era_genre(line):
	bandname = line.split(",")[0]
	url = "http://www.metal-archives.com/bands/" + bandname
	era_genre = line.split(",")[1:]
	era_genre = [l.replace(" ", "") for l in era_genre]
	era_genre[len(era_genre) - 1] = era_genre[len(era_genre) - 1].split("\n")[0]

	return bandname, url, era_genre

def choose_band(url):
	try:
		fp = urllib2.urlopen(url)
		html = fp.read()
		fp.close()

		if "may refer to" in html:	#if multiple candidates found
			soup = BeautifulSoup(html)
			bandlinks = [a.get("href") for a in soup.find_all("a") if "/bands/" in str(a)]
			prop_url = bandlinks[0]
		else:
			prop_url = url
		error_flag = 0

	except urllib2.URLError:
		error_flag = 1
		prop_url = "http://www.fake"

	return prop_url, error_flag


def discography_search(url, error_flag, failed_data, line):
	url_disc = "http://www.fake"

	if error_flag == 0:
		try:
			fp = urllib2.urlopen(url)
			html = fp.read()
			fp.close()
			soup = BeautifulSoup(html)

			link_all = soup.find_all(href=re.compile("discography"))
			for link in link_all:
				if str(link.string) == 'Complete discography':
					next_link = link
					url_disc = next_link.get("href")
					break

		except urllib2.URLError:
			print "Couldn't find the database for %s." % bandname.split("/")[0]
			error_flag = 1

		except:
			print "Unexpected Error for %s. No Discography?" % bandname.split("/")[0]
			error_flag = 1
	else:
		print "Error:Couldn't find the band info."
		failed_data += line
		error_flag = 1

	return url_disc, error_flag, failed_data


def process_era(num_album, era_genre):
	genre_list = list()

	if len(era_genre) == 2 and era_genre[0] == "0": # only 1 genre assigned
		genre = era_genre[1]
		for i in xrange(num_album):
			genre_list = [genre for i in xrange(num_album)]

	else:
		genre_list = ["dammy" for i in xrange(num_album)]
		num_era = len(era_genre) / 2
		era_genre.reverse()
		for i in xrange(num_era):
			era = era_genre.pop()
			if era[-1] == "-":
				era += str(num_album)
			elif era[0] == "-":
				era = "1" + era
			elif len(era) == 1:
				era = era + "-" + era
			start = int(era.split("-")[0])
			end = int(era.split("-")[1])
			#start, end = assign_genre(era, num_album)
			genre = era_genre.pop()
			for n in xrange(end - start + 1):
				genre_list[start - 1 + n] = genre		# trick
		genre_list[len(genre_list) - 1] = genre_list[len(genre_list) - 1].split("\n")[0]

	return genre_list


def assign_genre(era, num_album):
	start = int(era.split("-")[0])
	end = int(era.split("-")[1])

	return start, end


def access_album_rec(url_disc, bandname, era_genre, target):
	fp = urllib2.urlopen(url_disc)
	html = fp.read()
	fp.close()
	soup = BeautifulSoup(html)

	target_links, links_all = list(), list()
	links_all.extend(soup.find_all(class_="album"))
	links_all.extend(soup.find_all(class_="other"))

	for n in xrange(len(links_all) / 3):
		if links_all[3 * n + 1].get_text() in target:
			target_links.append(links_all[3 * n].get("href"))

	num_album = len(target_links)

	if num_album > 0:
		genre_list = process_era(num_album, era_genre)
	else:	# is it OK?
		genre_list = list()
		print "%s have never released albums. Process skipped." % bandname.split("/")[0]

	return target_links, genre_list, num_album


def get_album_titles(links, bandname):
	imgurls, titles, failed_set = list(), list(), set()

	for link in links:
		p_link, soup = choose_proper_link(link, bandname)
		if not p_link == "dammy":	# img exits
			imgsrc = soup.find("a", class_="image")
			imgurl = imgsrc.get("href")
			imgurls.append(imgurl)
			title = imgsrc.get("title")
			title = re.sub(r'[- *:/\\|?]', '_', title)
			title = title.replace("___", "_")
			titles.append(title)
		else:
			imgurls.append(links.index(link))
			titles.append("dammy")
			failed_set.add(links.index(link))

	return imgurls, titles, failed_set


def get_genre_vector(genres, genre, num_genre):
    genre_vector = np.zeros(num_genre)
    try:
        genre_vector[genres.index(genre)] = 1
    except:
        genre_vector[15] = 1

    return genre_vector


def choose_proper_link(link, bandname):
	fp = urllib2.urlopen(link)
	html = fp.read()
	fp.close()

	soup = BeautifulSoup(html)
	fmt = soup.find("dt", text="Format:").find_next().get_text()
	imgid = str(soup.find("a", class_="image"))
	if ("CD" in fmt or "vinyl" in fmt or "Digital" in fmt) and 'id="cover"' in imgid:
		p_link = link
	else:
		p_link = "dammy"
		links = soup.find("a",text="Other versions")
		if links is None:
			print "No CD or Vinyl found for %s" % link[37 + len(bandname.split("/")[0]) + 1:]
		else:
			fp = urllib2.urlopen(links.get("href"))
			html = fp.read()
			fp.close()
			soup_cand = BeautifulSoup(html)

			table = soup_cand.find_all("table")[0]
			rows = table.find_all("tr")[2:]
			row_elms = [row.find_all("td") for row in rows]
			format_list = [r[3].get_text() for r in row_elms]
			cd_vin_ind = np.where([l in ['CD', '12" vinyl', 'Digital', '7" vinyl'] for l in format_list])[0]

			if len(cd_vin_ind) != 0:
				all_links = np.array([l.get("href") for l in soup_cand.find_all(href=re.compile("albums"))[1:]])[cd_vin_ind]

				for link_cand in all_links.tolist():
					fp = urllib2.urlopen(link_cand)
					html = fp.read()
					fp.close()
					soup = BeautifulSoup(html)
					if 'id="cover"' in str(soup.find("a", class_="image").get_text):
						p_link = link_cand
						break
					else:
						continue
			else:
				print "CD or Vinyl eddition not found" # never happens

	return p_link, soup


def download(imgurl, bandname, title, num_cand, imgurls):
	if  imgurl != imgurls.index(imgurl) and type(imgurl) is not int:
		img = urllib.urlopen(imgurl)
		current_dir = os.getcwd()
		rt = current_dir + "/imgs/"
		path = rt + title + ".jpg"

		localfile = open(path, 'wb')
		localfile.write(img.read())
		img.close()
		localfile.close()
		im = scipy.misc.imread(path)
		os.remove(path)

		if len(im.shape) != 3 or ((im.shape[0] / float(im.shape[1])) > 1.4) or ((im.shape[0] / float(im.shape[1])) < 0.7) or im.shape[2] != 3:
			#print "%s has some problems. Not square or monochrome." % title
			im2 = imgurls.index(imgurl)
		else:
			im2 = scipy.misc.imresize(im, [224, 224])
			im2 = np.array(im2)
			sys.stdout.write("\r\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
			sys.stdout.write("\r%s worked!                  " % title)
			sys.stdout.flush()
	else:
		im2 = imgurls.index(imgurl)

	return im2




def get_target_ind(bandname, data):
	indices = [d[0] for d in data[bandname]]
	start = indices[0]
	end = indices[-1]
	return (start, end), data

def change_genre(changing_labels, new_label, genres, data, bandname):
	line = bandname + "," + new_label
	bandname, url, era_genre = align_era_genre(line)
	print era_genre
	if len(era_genre) == 2 and era_genre[0] == "0":
		changed_labels = [np.zeros(num_genre) for i in xrange(len(changing_labels))]
		new_ind = genres.index(era_genre[1])
		for changed_label in changed_labels:
			changed_label[new_ind] = 1

		for d in data[bandname]:
			d[2] = era_genre[1]

	else:	# new_label = "1-4,Thrash,5-,Groove" etc
		last = era_genre[-2]
		if last[-1] == "-":
			last += str(num_album)
		elif last[0] == "-":
			last = "1" + last
		elif len(last) == 1:
			last = last + "-" + last

		assert last[-1] >= len(changing_labels), "Error"


		changed_labels = [np.zeros(num_genre) for i in xrange(len(changing_labels))]
		num_era = len(era_genre) / 2
		era_genre.reverse()
		count = 0
		for i in xrange(num_era):
			add_count = count
			era = era_genre.pop()
			if era[-1] == "-":
				era += str(len(changing_labels))
			elif era[0] == "-":
				era = "1" + era
			elif len(era) == 1:
				era = era + "-" + era
			start = int(era.split("-")[0])
			end = int(era.split("-")[1])

			genre = era_genre.pop()
			for n in xrange(end - start + 1):
				changed_labels[n + add_count][genres.index(genre)] = 1
				data[bandname][n + add_count][2] = genre
				count += 1

	return changed_labels, data


genres = ["Heavy", "Hair", "Thrash", "Doom", "Death", "Black",
          "Symphonic", "Power", "Progressive", "Metalcore",
          "Grindcore", "Groove", "Goregrind", "Gothic", "Viking", "False"]

target = ["EP", "Full-length", "Single"]	# excluded "Live album"

url_tmp = "http://www.metal-archives.com/bands/"
num_genre = len(genres)

with open("failed_s.txt", 'r') as f_f:
	no_info = f_f.readlines()

no_info = set(no_info)

images, labels = list(), list()
data = dict()
worked_data = ""	# stored as string
failed_data = ""	# stored as string

with open("new_input.txt", 'r') as f:
	line = f.readline()

	count = 0
	while line:
		error_flag = 0
		bandname, url, era_genre = align_era_genre(line)
		print "Processing %s..." % bandname.split("/")[0]

		if len(era_genre) % 2 != 0 or line in no_info:
			failed_data += line
			print "invalid input data: %s" % bandname.split("/")[0]

		else:
			try:
				proper_url, error_flag = choose_band(url)
				url_disc, error_flag, failed_data = discography_search(proper_url, error_flag, failed_data, line)
				target_links, genre_list, num_album = access_album_rec(url_disc, bandname, era_genre, target)
				if num_album == 0:
					"Skipped %s. Error detected." % bandname.split("/")[0]
					failed_data += line
					error_flag = 1
					line = f.readline()
					continue

				imgurls, titles, failed_record_set = get_album_titles(target_links, bandname)

				cover_imgs = [download(imgurls[j], bandname, titles[j], num_album, imgurls) for j in xrange(len(imgurls))]
				failed_img_set = set([x for x in cover_imgs if type(x) is int])

				genre_vectors = [get_genre_vector(genres, genre_list[j], num_genre) for j in xrange(num_album)]
				failed_img_set.update(failed_record_set)

				images.extend([cover_imgs[i] for i in xrange(len(cover_imgs)) if i not in failed_img_set])
				labels.extend([genre_vectors[i] for i in xrange(len(genre_vectors)) if i not in failed_img_set])

				titles = [titles[i] for i in xrange(len(titles)) if i not in failed_img_set]

				if error_flag == 0:
					data[bandname.split("/")[0]] = [(n, titles[n - count], genre_list[n - count]) for n in xrange(count, count + len(titles))]
					count += len(titles)
					worked_data += line
					print "%s finished." % bandname.split("/")[0]
				else:
					failed_data += line
			except:
				"Skipped %s. Error detected." % bandname.split("/")[0]

		line = f.readline()

assert len(images) == len(labels), "error:close!" # is it good?

images = np.array(images)

data["Summary"] = list()

count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for l in labels:
	count += l

for i in xrange(len(count)):
	print "%s - %d images" % (genres[i], count[i])
	data["Summary"].append((genres[i], count[i]))
print "Total - %d" % np.array(count).sum()
data["Summary"].append(("Total", np.array(count).sum()))

# create output h5 file
N = len(images)
out = "metalcovers_l.h5"
f = h5py.File(out, "w")
f.create_dataset("images", dtype='uint8', data=images)
f.create_dataset("labels", dtype='uint32', data=labels)
f.close()


data_json = json.dumps(data)

with open("worked_l.txt", "w") as f_w:
	f_w.write(worked_data)

with open("failed_l.txt", "w") as f_f:
	f_f.write(failed_data)

with open("band_label_l.json", "w") as f_j:
	f_j.write(data_json)





"""

# Usage:
bandname = "Metallica"
data[bandname]

tuple, data = get_target_ind(bandname, data)
labels[tuple[0]:tuple[1]], data = change_genre(labels[tuple[0]:tuple[1]], "0, Thrash", genres, data, bandname)

labels[tuple[0]:tuple[1]], data = change_genre(labels[tuple[0]:tuple[1]], "1-4, Thrash,5,Heavy,6-7,False, 8-,Thrash", genres, data, bandname)
"""