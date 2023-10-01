select case
	when "Marital Status" = '' then 'Other'
	else "Marital Status" end
"Marital Status",
avg(age) as avg_age from customer
group by "Marital Status";


--select case
--	when gender = 0 then 'Woman'
--	else 'Man' end
--gender,
--avg(age) as avg_age from customer
--group by gender;


--select s.storename,
--sum(t.qty) as sum_qty from store as s
--inner join "transaction" as t
--on s.storeid = t.storeid
--group by s.storeid, s.storename
--order by sum_qty desc
--limit 1;


--select p."Product Name",
--sum(t.totalamount) as sum_totalamount from product as p
--inner join "transaction" as t
--on p.productid = t.productid
--group by p.productid, p."Product Name"
--order by sum_totalamount desc
--limit 1;
