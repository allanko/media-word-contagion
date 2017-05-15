# Project

We're looking at the [US Presidential Election](https://topics.mediacloud.org/#/topics/1404/summary) topic in [Media Cloud](https://mediacloud.org/). That's topic ID #1404. This is a set of stories published between Apr 30, 2015 to Nov 7, 2016, queried on the names of the major presidential candidates. The topic is queried from the following media source sets:

* [US Top Online News](https://sources.mediacloud.org/#/collections/9139487)
* [US Top Digital Native News](https://sources.mediacloud.org/#/collections/9139458) 
* [US Regional Mainstream Media](https://sources.mediacloud.org/#/collections/2453107) 

The seed query is:

> +( fiorina ( scott and walker ) ( ben and carson ) trump ( cruz and -victor ) kasich rubio (jeb and bush) clinton sanders ) AND (+publish_date:[2016-09-30T00:00:00Z TO 2016-11-08T23:59:59Z]) AND ((tags_id_media:9139487 OR tags_id_media:9139458 OR tags_id_media:2453107 OR tags_id_stories:9139487 OR tags_id_stories:9139458 OR tags_id_stories:2453107))

I *think* this is the same dataset used for this CJR report, ["Breitbart-led right-wing media ecosystem altered broader media agenda"](https://www.cjr.org/analysis/breitbart-media-trump-harvard-study.php), but I'm not totally sure.

# Setup

Make sure you have the [Media Cloud API client](https://github.com/mitmedialab/MediaCloud-API-Client) installed. Also make sure you have an account - set it up on the [Media Cloud website](https://mediacloud.org/).

Then, make a copy of `app.config.template` and fill it in with your API key. Set the TOPIC_ID as 1404. Rename the file `app.config`.
