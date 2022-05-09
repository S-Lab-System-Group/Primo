
### Primo + Pensieve Deployment

+ **Tested OS**: Ubuntu 20.04

+ **Prerequisite Environment**:  

| Software  | Version | Install Command                             |
|-----------|---------|---------------------------------------------|
| apache    | 2.4.41  | `sudo apt install apache2`                  |
| node.js   | 10.19.0 | `sudo apt install nodejs`                   |
| npm       | 6.14.4  | `sudo apt install npm` (bundle with nodejs) |
| grunt-cli | 1.0.1   | `npm install -g grunt-cli@1.0.1`            |
| dash.js   | 2.3.0   | See below                                   |


### After installing environment, steps for generating `dash.all.min.js`:

1. `cd deploy/dash.js` 

2. `npm install` (ignore warning & errors)

3. `grunt --force` (ignore warning & errors)

4. Enter folder `dash.js/dist/`, copy `dash.all.min.js` and `dash.all.min.js.map`(optional) to  `./video_server`.

5. We have provide ***video1~video6*** folders (from [pensieve repo](https://github.com/hongzimao/pensieve/tree/master/video_server)) in `./video_server`. You can replace with your videos.

6. Move the `./video_server/` to `/var/www/html/`.

7. Visit the `http://localhost/index_XX.html` (XX should be the name of the ABR). The memory and latency statistics will be displayed on the web page. (Note viper represents Metis.)

8. (Optional) You can refer `py2js.md` to transfer customized learned Primo model into js format and revised `deploy/dash.js/src/streaming/controllers/PrimoDecisionTree.js` accordingly.
