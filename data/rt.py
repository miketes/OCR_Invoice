import xlwt
import sql as sql
import numpy as np

def write_xls():
    data = sql.Select()
    data = np.array(data)

    lists = ['发票类型','发票号码','发票代码','销售方名称','销售方识别号','销售方地址和电话','销售方银行信息',
              '购货方名称', '购货方识别号','购货方地址和电话','购货方银行信息','商品信息(品名/规格/产地','单位',
              '数量','单价','金额','税率','税额','发票总额','规格型号','效验码','开票日期','收款人','复核人','开票人',
              '备注','金额合计','税额合计','验真时间']

    nm = 0
    ID = 0
    lens = []

    for i in data:
        i = np.delete(i, (0, 29, 31,32), 0)
        for k in i:
            ds = k.split('==')
            le = len(ds)
            lens.append(le)

        lens.sort(reverse=True)


        # print(lens[0])
        book = xlwt.Workbook()
        sheet = book.add_sheet('sheet1')
        ID += 1
        ad = i[13].split('==')

        nm = 0
        for j in range(0,len(i)):
            xx = i[j].split('==')

            sheet.write(0, nm, lists[j])


            for z in range(0,len(xx)):
                # print(xx)
                if len(xx) == 1:
                    # print(z+1, nm, xx,'=======================')
                    for s in range(0,lens[0]):
                        sheet.write(s+1, nm, xx)
                    # sheet.write(z+1,nm, xx[z])
                else:
                    # print(z+1,nm,xx,'----------------------')
                    sheet.write(z + 1, nm, xx[z])
            nm += 1

        print(ID,'=')
        book.save('../识别结果/'+ i[1] +'.xls')


write_xls()